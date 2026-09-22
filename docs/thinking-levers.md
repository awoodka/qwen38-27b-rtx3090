# Thinking levers

This fork adds switches that steer how Qwen3.8-27B spends its thinking, without retraining
it and without changing anything while they are off. This page covers the one that exists
so far, the reasoning prompt (`REASONING_EFFORT`), and how the levers are judged.

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

## Lever 2: a penalty on reflection markers (in progress)

A vLLM patch that lowers the logits of a few reflection markers ("Wait", "Hmm",
"Alternatively") while the model is inside `<think>`, and works with DFlash2 speculative
decoding. It is the idea of NoWait ([arXiv 2506.08343](https://arxiv.org/abs/2506.08343))
applied without fine-tuning. Expect a modest effect: Qwen3.8 rarely uses these words, with
"Wait" at 1.36 and "Hmm" at 0.09 per 1,000 words of the baseline's LiveCodeBench reasoning.

## How the levers are judged

Quality comes first. Each lever runs the baseline's evals and is compared with it task by
task. A lever succeeds if:

- no benchmark's paired accuracy drops by more than one standard error,
- fewer answers run out of budget,
- the reasoning loops on markers less, and
- decode speed stays within 3% of the server with the levers off.

If neither lever helps, this page will say so. Results: pending.
