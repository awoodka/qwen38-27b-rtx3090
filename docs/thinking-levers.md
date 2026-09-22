# Thinking levers

This fork adds switches that steer how Qwen3.8-27B spends its thinking, without retraining
it and without changing anything while they are off. This page covers both, the reasoning
prompt (`REASONING_EFFORT`) and the marker penalty (`THINK_PENALTY`), and how they are
judged.

## Why

Qwen3.8 can think for a very long time. In the quick-tier eval of this stack that the
levers are measured against ([run page](https://localinference.alexwoodka.com/runs/83638382-55b5-460b-a309-e80986a75c7d):
LiveCodeBench, GPQA Diamond, AIME 2025 and BFCL on `SPEC=dflash2` at 250 W, with a
5-minute budget per task, about 42,000 output tokens at this stack's speed), 73 answers
ran out of budget: 32 of 100 LiveCodeBench tasks, 23 of 198 GPQA questions and 18 of 120
AIME attempts. The answers that finished passed at 88%, 93% and 97%. A cut-off scores as
wrong, so turning cut-offs into finished answers is where the quality is.

The answer is not to think less across the board. Swift-Qwen3.8-27B, a LoRA fine-tune that
penalizes reasoning-marker tokens, cuts 25-50% of the thinking and loses 3-5 points on
AIME and HMMT. The levers aim at the thinking that does not move an answer forward.

## Lever 1: the reasoning prompt (`REASONING_EFFORT`)

### What it changes

Whenever thinking is on, the model's chat template puts a reasoning instruction in the
system turn. At the default level, xhigh, it reads:

> Reasoning effort is set to xhigh. Please think carefully through the task, validate key
> assumptions, consider plausible alternatives, and prioritize correctness, consistency,
> and clarity in the final answer.

"Validate key assumptions, consider plausible alternatives" invites the re-checking that
runs long. The fork's template, [`templates/qwen3.8-27b.jinja`](../templates/qwen3.8-27b.jinja),
adds a level called `focused` that asks for the same depth without it:

> Reasoning effort is set to xhigh. Correctness comes first, so think as carefully as the
> task requires, but make every step move the solution forward. Compare approaches
> briefly, choose one, and commit to it. Check each result once, when you derive it; once
> it checks out, treat it as settled. Do not re-verify settled steps, restart from scratch,
> or reopen a choice you have already made unless you find a specific error; if you find
> one, fix it where it occurred and continue from there. When you are confident in the
> answer, stop thinking and give the final answer.

About the wording:

- The first sentence is xhigh's, the cue the model presumably learned depth from, so the
  depth signal is unchanged. A variant that opens "Reasoning effort is set to focused." is
  the alternative to compare before the evals, and this page will record which text was
  measured.
- It explicitly allows fixing a real error. Many of Qwen3.8's "Wait"s are genuine error
  catches, and those should survive.
- It has no apostrophes, because it sits in a single-quoted Jinja string.

The template also accepts four names from the OpenAI vocabulary that the original rejects
with an exception, which a client sees as a 400: `max` and `high` are served as xhigh,
and `minimal` and `none` as low. A top-level `reasoning_effort: "none"` never reaches that
check, because vLLM turns thinking off for it. Any other value the original rejects, the
fork rejects too, and for every input the original accepts, the fork renders the same
bytes.

### Using it

| `REASONING_EFFORT` | Template served | Server default |
|---|---|---|
| unset or empty | the model's own | xhigh (the template's own default) |
| `xhigh` | the fork's | xhigh: the same prompts as unset, but a request can ask for `focused` |
| `focused` | the fork's | focused |
| `medium` or `low` | the fork's | that level |

```bash
REASONING_EFFORT=focused SPEC=dflash2 PREFIX_CACHE=1 bash single-user/start_qwen.sh
```

A request's own effort wins over the server default. From strongest to weakest:

1. The top-level `reasoning_effort`, OpenAI's field: none, minimal, low, medium, high,
   xhigh or max. It has no `focused`.
2. `"chat_template_kwargs": {"reasoning_effort": "focused"}`. This is the only way to ask
   for `focused` per request, and it needs the fork's template, so any non-empty
   `REASONING_EFFORT`.
3. The server default that `REASONING_EFFORT` sets.

vLLM's `/tokenize` applies the same default, so client-side prompt counts stay exact.

The launcher adds `--chat-template templates/qwen3.8-27b.jinja` and
`--default-chat-template-kwargs '{"reasoning_effort":"<level>"}'` right after
`--reasoning-parser qwen3` and before `EXTRA_ARGS`, which can still override both. Nothing
else in the command line changes. Only the venv path has the knob: the Docker quick start
pulls upstream's prebuilt image, which does not.

### Checks

- `bench/test_chat_template.py`, on the CPU in about 6 s:
  `CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_chat_template.py`. It renders 216 inputs
  (with and without a system message or tools, thinking unset, on and off,
  `preserve_thinking` unset, on and off, a multi-turn conversation with reasoning and a tool
  call, with and without a generation prompt) at every effort the original accepts:
  1,296 renders that must match the original byte for byte. `focused` must equal the xhigh
  render with the instruction swapped, the aliases must equal their levels, unknown efforts
  and malformed conversations must still raise, and the template variables vLLM uses to
  filter `chat_template_kwargs` must not change.
- `bench/test_launcher_args.sh`, no GPU: `bash bench/test_launcher_args.sh`. With the knob
  unset, the launcher's argv and output must equal bae2023's across ten configurations.
  Each level must add exactly the two flags, and anything else must exit 1 before vLLM
  starts.
- A live check on an RTX 3090 at 250 W, 2026-09-22, with
  `REASONING_EFFORT=focused SPEC=dflash2 PREFIX_CACHE=1`. `/tokenize` counted what the
  server billed, with and without tools (153 and 406 prompt tokens). All seven top-level
  efforts returned 200 and billed their level's prompt: 68 tokens for high, xhigh and max,
  56 for minimal and low, 26 for medium and 28 for none (thinking off), against 153 for
  the focused default. A bogus level in `chat_template_kwargs` returned a 400 that names
  the levels. The model thought and answered under focused, a tool call parsed, and
  `bench/api_smoke.py` passed 12/12.

### Upstream fixed the same 400 differently

After bae2023, upstream added `prepare/translate_chat_template.py` (their gotcha 58,
PR #124), which rewrites the model's own template in Docker's prepare step: minimal becomes
low, high and max become xhigh, and any other unknown value gets no instruction instead of
an exception. The fork's template agrees on those three names. It also maps none to low,
and it keeps the exception for names it does not know, so a typo fails loudly. It leaves
the model's template on disk alone, and it is only served when `REASONING_EFFORT` is set.

## Lever 2: a penalty on reflection markers (`THINK_PENALTY`)

### What it changes

While the model is inside its reasoning, vLLM subtracts `THINK_PENALTY` logits from the
tokens of a few reflection markers: "Wait", "Hmm" and "Alternatively" by default, each as
written and with a leading space, so six tokens. This is the idea of NoWait
([arXiv 2506.08343](https://arxiv.org/abs/2506.08343)) applied without fine-tuning. The
penalty comes before temperature, so the effect on the probabilities is λ/T (T is 1 at the
model's default sampling), and before top-k/top-p, so a marker that was not clearly the
model's choice drops out of the nucleus. At λ = 100 the markers are banned in practice,
which is NoWait's setting.

"Inside its reasoning" means that in the tokens before the position being scored, the last
`<think>` comes after the last `</think>`. The answer after `</think>` is never penalized,
and neither is anything with thinking off, a raw completion without markers, or an earlier
turn of the conversation. A request that also sets `thinking_token_budget` gets both, and
the budget still forces `</think>` when it runs out.

It has to be a vLLM patch. Under speculative decoding vLLM rejects `logit_bias` and custom
logits processors, `bad_words` would remove the words from code answers too, and the
server has no way to default any of them. [`patches/think-penalty.patch`](../patches/think-penalty.patch)
touches three files that no other patch touches:

- `config/reasoning.py`: `think_penalty` and `think_penalty_words`, resolved to token ids at
  startup. The kept and dropped forms are logged; a form must be one ordinary token (not a
  special token, not a reasoning marker). An unpatched vLLM rejects the keys outright.
- `v1/worker/gpu/sample/thinking_budget.py`: Model Runner V2 already tracks, per request,
  where the last `<think>` and `</think>` are, including in the draft tokens each verify row
  sees, but only for requests with a thinking budget. With the penalty on, a request without
  a budget gets one it can never reach, so the tracking runs for every request and a reused
  slot starts clean. A small Triton kernel after the budget kernels then penalizes each
  logits row that is inside reasoning. Every verify row of a speculative step goes through
  it, so DFlash2's output follows the penalized distribution exactly; the drafter is not
  penalized, which can cost a little acceptance.
- `config/vllm.py`: Model Runner V1 (`SPEC=mtp`) refuses the setting instead of silently
  ignoring it.

### Using it

```bash
THINK_PENALTY=3 SPEC=dflash2 PREFIX_CACHE=1 bash single-user/start_qwen.sh
THINK_PENALTY=3 THINK_PENALTY_WORDS=Wait,Hmm,Alternatively,Actually SPEC=dflash2 bash single-user/start_qwen.sh
```

Measured below: only `THINK_PENALTY=100`, a ban in practice, shortens the thinking; lower
values remove the words and the model says others instead.

The launcher passes `--reasoning-config
'{"think_penalty":3,"think_penalty_words":["Wait","Hmm","Alternatively"]}'`, and the server
logs `Think penalty: -3.00 logits on 6 token ids inside reasoning`. It is a server setting
that applies to every request inside its reasoning, with no per-request control, and it
combines with `REASONING_EFFORT`. `THINK_PENALTY` runs from 0 (off) to 100, and the launcher
refuses anything else, a word that is not letters and hyphens, and any `SPEC` other than
`dflash2`.

### The word list, from the baseline's reasoning

In the baseline's LiveCodeBench reasoning (1.34 million words), sentence-initial "Wait"
occurs 1.3 times per 1,000 words, "Actually" 1.1, "Hmm" 0.1 and "Alternatively" 0.04; in
BFCL, "Hmm" is at 0.95. Per answer, "Wait", "Hmm" and "Alternatively" together come to 1.5
per 1,000 words in the LiveCodeBench answers that were cut off, against 1.1 in the ones that
finished, and "Maybe" to 1.3 against 0.6. Pooled over all words, the three barely differ
(1.50 against 1.43), and many "Wait"s are genuine error catches. So expect a modest effect.
The default stays the classic three, NoWait's kind of marker, and "Actually" and "Maybe"
are screened on top of it.

### Checks

- `bench/test_think_penalty.py --cpu`: the config's validation; the six default ids with
  the real tokenizer and qwen3 parser; multi-token forms, markers and special tokens
  dropped. It runs the kernels in Triton's interpreter against a pure-Python reference: 14
  named cases (inside reasoning, `</think>` as a middle draft, a committed `</think>`, a
  thinking-off prompt, no markers, a reopened `<think>`, multi-turn history, a cold scan
  across 1,024-token blocks, an exhausted budget plus the penalty, a mixed batch), a stale
  marker cache, and 568 random multi-step decode scenarios. With the penalty off, logits
  stay bitwise unchanged. Five deliberately broken kernels were each caught.
- `bench/test_think_penalty.py --gpu`, under `lab gpu run`: the same cases with compiled
  kernels, plus `ThinkingBudgetState` itself. The penalty reaches requests without a
  budget, a reused slot starts clean for a shorter and a longer request, a real budget
  still forces `</think>`, penalty 0 leaves logits bitwise unchanged, and the log line
  appears. All 645 checks passed on the RTX 3090 on 2026-09-22.
- `bench/test_launcher_args.sh`: the exact `--reasoning-config` for each setting, nothing at
  0, and the refusals.
- The live session (`bench/thinking_levers_session.sh`), whose results are below: the marker rates
  it measures, the answers keeping the penalized words, the budget still capping reasoning, and the
  speed comparison at λ = 3.

## First measurements (2026-09-22)

Two unattended sessions on an RTX 3090 at 250 W, `SPEC=dflash2 PREFIX_CACHE=1`, served from this
fork's pinned runtime. These are server-level checks, not eval results: they say what each lever
does to the model's thinking, not whether the answers get better. The evals decide that, and they
have not run yet. Raw output lands in `bench/results/levers-*`, which git ignores.

### With the levers off, this is upstream

- Speed matches: 125.9 tok/s decode at C1 with default sampling and 137.6 greedy, at 3.24 and 3.48
  tokens per step (`run_benchmarks.sh single`, the second run after the restart). The same card
  measured 125.8 / 124.4 and 137.4 / 136.8 on bae2023 a week earlier.
- Greedy output is reproducible within one installation: restarting the server reproduces all 12
  prompts exactly, for bae2023 and for this fork alike.
- Across installations it is not, with or without this fork. A second copy of bae2023's vLLM, byte
  for byte identical and installed beside it, agreed with it on 5 of 12 prompts; the fork with the
  levers off also agrees on 5 of 12 with bae2023, and on 6 of 12 with that unpatched second copy.
  Every divergence is mid-answer at a near-tie, "closely" against "tightly" or "divisors" against
  "factors". Swapping the FlashInfer JIT build changed nothing, so each installation settles on its
  own kernel numerics. The fork therefore differs from upstream no more than reinstalling upstream
  does, which is also why every arm of an experiment should run from one installation.

### The reasoning prompt

`/tokenize` counted exactly what the server billed (153 prompt tokens, and 406 with tools), the
rendered system turn carried the focused text, and all seven OpenAI effort levels answered 200,
where the model's own template answers 400 to high, minimal and max. On thinking length it did
nothing measurable: paired against the levers-off server over 24 answers to the harder prompts, the
ratio is 0.97 with a 95% interval of [0.80, 1.16], the marker rates are unchanged, and the same
three answers ran out of room.

### The token penalty

Twelve harder prompts (`bench/prompts_thinking_hard.jsonl`), two samples each, the same seed per
prompt and sample in every configuration, paired against the levers-off server
(`bench/thinking_levers.py paired --out-dir <results dir>` prints this table, with a bootstrap
interval over the pairs):

| `THINK_PENALTY` | reasoning tokens vs off [95% CI] | markers per 1k reasoning tokens | ran out of room | tokens per step |
|---|---|---|---|---|
| off | — (5,552 mean, 2,453 median) | 1.32 | 3 of 24 | 4.01 |
| 1.5 | 1.17 [0.87, 1.54] | 0.66 | 4 | 3.82 |
| 3 | 1.07 [0.87, 1.36] | 0.27 | 5 | 3.88 |
| 6 | 0.93 [0.78, 1.11] | 0.04 | 3 | 3.88 |
| 100 (a ban) | **0.77 [0.63, 0.94]** | 0.00 | 2 | 4.08 |

- The penalty removes what it aims at: "Wait" falls from 1.21 per 1,000 reasoning tokens to 0.62,
  0.26, 0.04 and 0, as λ rises.
- Only a ban shortens the thinking. At λ = 100 reasoning is about a quarter shorter, the interval
  excludes 1, fewer answers run out of room, and tokens per step do not suffer. Below that the model
  keeps thinking just as long and reaches for other words: at λ = 1.5, "Actually" rises from 0.65 to
  0.85 and "Maybe" from 0.82 to 1.08.
- Screening one more word at λ = 3 did not change that: adding "Actually" gives 1.20 [0.90, 1.60]
  and adding "Maybe" 0.89 [0.65, 1.23], each suppressing its own word. Neither joins the default
  list.
- The answers keep the words at every λ: "Reply with exactly this text: Wait, Hmm, Alternatively,
  Actually, Maybe." came back verbatim, and code still called `asyncio.wait` and `Condition.wait()`.
- A request's own `thinking_token_budget` still wins with the penalty on: a budget of 32 capped
  reasoning at 31 tokens.
- Speed: greedy decode at C1 is 138.1 tok/s against 137.6 with the levers off (3.50 against 3.48
  tokens per step), so the kernel itself costs nothing measurable. At default sampling one run each
  read 120.9 against 125.9 tok/s, −4.0%, which is inside the spread of the runs themselves: the two
  λ = 3 runs differed by 4.4% and the C2 cohort went the other way by 5.6%. Resolving 3% at default
  sampling needs repeated runs.

### What that means for the evals

The mechanism works as designed, and on this model the setting worth evaluating is the ban, which
is the shape NoWait reports. Whether a quarter less thinking costs accuracy is exactly what the
paired evals have to answer, and Swift's fine-tune is the warning: it lost 3 to 5 points on hard
math for a similar cut. So the pilot runs `focused` and λ = 100, with λ = 6 as a near-ban control,
and not the soft penalties.

## How the levers are judged

First, a live session on the GPU, `bench/thinking_levers_session.sh` (about 2 hours,
unattended), checks each lever as a server setting: the fork with both levers off answers
greedy prompts like bae2023 does across a restart, focused renders and bills correctly, the
penalty lowers the marker rates in reasoning as λ rises while the answers keep those words,
and at λ = 3 decode speed stays within 3% and tokens per step within 0.1 of the levers-off
server. It also screens "Actually" and "Maybe" on top of the default words. Its client,
`bench/thinking_levers.py`, runs the twelve original prompts in `bench/prompts_thinking.jsonl`
and writes `summary.md`.

Then quality, which comes first. Each lever runs the baseline's evals and is compared with
it task by task. A lever succeeds if:

- no benchmark's paired accuracy drops by more than one standard error,
- fewer answers run out of budget,
- the reasoning loops on markers less, and
- decode speed stays within 3% of the server with the levers off.

If neither lever helps, this page will say so. The server-level measurements are above; the eval
results are still to come.
