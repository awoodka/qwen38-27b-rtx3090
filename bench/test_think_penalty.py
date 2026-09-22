"""Tests for patches/think-penalty.patch (the fork's THINK_PENALTY lever).

The patch subtracts `think_penalty` from the logits of a few reflection-marker tokens while a
request is inside its reasoning, in Model Runner V2's sampler, on every row a speculative verify
step scores. This checks it against a from-first-principles reference:

  --cpu  the config's validation, the word -> token id resolution with the model's real tokenizer
         and qwen3 reasoning parser, and the kernels in Triton's interpreter: the named cases below
         plus a few hundred random scenarios, one and several decode steps each.
  --gpu  the same kernel cases compiled on the GPU, plus ThinkingBudgetState itself: the penalty
         reaches every request, a reused slot never carries the previous request's markers, a real
         thinking_token_budget still wins, penalty 0 leaves logits bitwise unchanged, and the log
         line appears. The card must be free: run it under `lab gpu run` next to a hosted model.

  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_think_penalty.py --cpu [--model DIR]
  venv/bin/python bench/test_think_penalty.py --gpu [--model DIR]

A row is inside reasoning when, in the tokens before the position it scores (the committed ones
plus the draft tokens that row sees), the last reasoning start marker comes after the last natural
end marker. Row p of a request verifying drafts d1..dk sees d1..dp.
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

MODE = "--gpu" if "--gpu" in sys.argv else "--cpu"
if MODE == "--cpu":
    # Before triton is imported: interpret the kernels on the CPU, keep off the GPU.
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

from vllm.config.reasoning import ReasoningConfig, resolve_think_penalty_token_ids  # noqa: E402
from vllm.v1.worker.gpu.sample.thinking_budget import (  # noqa: E402
    _INT32_MAX,
    apply_think_penalty,
    apply_thinking_budget,
)

REPO = Path(__file__).resolve().parent.parent
THINK, END_THINK = 248068, 248069          # <think>, </think> in the Qwen3.8 vocabulary
VOCAB = 248320
DEFAULT_WORDS = ["Wait", "Hmm", "Alternatively"]
DEFAULT_IDS = [13428, 13784, 37201, 77264, 85152, 88842]   # " Wait" "Wait" " Alternatively" "Hmm" " Hmm" "Alternatively"
DEVICE = "cuda" if MODE == "--gpu" else "cpu"

fails = 0
counts: dict[str, list[int]] = {}


def check(name, ok, detail=""):
    global fails
    n = counts.setdefault(name, [0, 0])
    n[0] += bool(ok)
    n[1] += 1
    if not ok:
        fails += 1
        if n[1] - n[0] <= 3:
            print(f"FAIL {name}: {detail}")


# -- the reference ------------------------------------------------------------------------------

def last_match(seq, marker):
    for i in range(len(seq) - len(marker), -1, -1):
        if seq[i:i + len(marker)] == marker:
            return i
    return -1


def reference(logits, batch, start, natural_end, forced_end, penalty_ids, penalty):
    """Expected logits after the budget step and the penalty step, row by row."""
    out = logits.clone()
    row = 0
    for req in batch:
        for p in range(len(req["drafts"]) + 1):
            seq = req["committed"] + req["drafts"][:p]
            ls, le = last_match(seq, start), last_match(seq, natural_end)
            if req["budget"] >= 0 and ls >= 0 and ls > le:
                if len(seq) - (ls + len(start)) >= req["budget"]:
                    prefix = 0
                    for k in range(1, len(forced_end)):
                        if k <= len(seq) and seq[len(seq) - k:] == forced_end[:k]:
                            prefix = k
                    out[row, forced_end[prefix]] = 1.0e9
                if penalty:
                    ids = torch.tensor(penalty_ids, dtype=torch.long)
                    out[row, ids] = out[row, ids] - torch.tensor(penalty, dtype=torch.float32)
            row += 1
    return out


# -- running the patched kernels the way ThinkingBudgetState.apply does -------------------------

class Slots:
    """The per-slot state the kernels read: token ids, lengths, budgets and the marker cache."""

    def __init__(self, n, max_len):
        self.all_ids = torch.zeros((n, max_len), dtype=torch.int32, device=DEVICE)
        self.total_len = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        self.budget = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
        self.last_start = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
        self.last_end = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
        self.scan_pos = torch.zeros(n, dtype=torch.int32, device=DEVICE)

    def add(self, slot, committed, budget):
        """A new request in `slot`: add_request + apply_staged_writes (the cache reset)."""
        self.all_ids[slot].zero_()
        self.all_ids[slot, :len(committed)] = torch.tensor(committed, dtype=torch.int32)
        self.total_len[slot] = len(committed)
        self.budget[slot] = budget
        self.last_start[slot], self.last_end[slot], self.scan_pos[slot] = -1, -1, 0

    def commit(self, slot, committed):
        self.all_ids[slot, :len(committed)] = torch.tensor(committed, dtype=torch.int32)
        self.total_len[slot] = len(committed)


def run(slots, batch, logits, start, natural_end, forced_end, penalty_ids, penalty):
    input_ids, exp_map, local = [], [], []
    for req in batch:
        input_ids += [req["committed"][-1]] + req["drafts"]
        exp_map += [req["slot"]] * (len(req["drafts"]) + 1)
        local += list(range(len(req["drafts"]) + 1))
    t = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEVICE)  # noqa: E731
    out = logits.clone().to(DEVICE)
    args = dict(input_ids=t(input_ids), expanded_local_pos=t(local))
    apply_thinking_budget(out, t([req["slot"] for req in batch]), t(exp_map), slots.budget, slots.all_ids,
                          slots.total_len, args["input_ids"], args["expanded_local_pos"], slots.last_start,
                          slots.last_end, slots.scan_pos, t(start), t(natural_end), t(forced_end))
    if penalty:
        apply_think_penalty(out, t(exp_map), slots.budget, slots.all_ids, slots.total_len, args["input_ids"],
                            args["expanded_local_pos"], slots.last_start, slots.last_end, t(start),
                            t(natural_end), t(penalty_ids), penalty)
    return out.cpu()


def rows_changed(before, after):
    return [r for r in range(before.shape[0]) if not torch.equal(before[r], after[r])]


# -- named cases, real token ids ------------------------------------------------------------------

def named_cases():
    """(name, batch of requests, rows that must be penalized). Budgets are the sentinel unless set."""
    W, H, A = 13428, 85152, 37201                  # " Wait", " Hmm", " Alternatively"
    x = [1000 + i for i in range(40)]              # ordinary tokens
    prompt = [151644, 872, 198] + x[:6] + [151645, 198, 151644, 77091, 198]
    think = prompt + [THINK, 198]                 # generation prompt with thinking on
    off = prompt + [THINK, 271, END_THINK, 271]   # enable_thinking=false
    S = _INT32_MAX
    return [
        ("in-think, 4 drafts", [dict(slot=0, committed=think + x[:9] + [W] + x[9:12], drafts=x[12:16], budget=S)], [0, 1, 2, 3, 4]),
        ("</think> as draft 3 of 4", [dict(slot=1, committed=think + x[:5], drafts=[x[5], x[6], END_THINK, 271], budget=S)], [0, 1, 2]),
        ("</think> as draft 1", [dict(slot=1, committed=think + x[:5], drafts=[END_THINK, 271, x[7]], budget=S)], [0]),
        ("committed </think>", [dict(slot=2, committed=think + x[:8] + [END_THINK, 271] + x[8:12], drafts=x[12:15], budget=S)], []),
        ("thinking-off prompt", [dict(slot=0, committed=off + x[:6], drafts=x[6:10], budget=S)], []),
        ("no markers at all", [dict(slot=3, committed=x[:20], drafts=x[20:24], budget=S)], []),
        ("reopened <think>, committed", [dict(slot=0, committed=think + x[:4] + [END_THINK] + x[4:6] + [THINK] + x[6:9], drafts=x[9:11], budget=S)], [0, 1, 2]),
        ("reopened <think>, as draft 2", [dict(slot=0, committed=think + x[:4] + [END_THINK, 271], drafts=[x[4], THINK, x[5]], budget=S)], [2, 3]),
        ("multi-turn history, new turn thinking", [dict(slot=2, committed=prompt + [THINK] + x[:3] + [END_THINK] + x[3:5] + prompt + [THINK, 198] + x[5:9], drafts=x[9:12], budget=S)], [0, 1, 2, 3]),
        ("multi-turn history, new turn not thinking", [dict(slot=2, committed=prompt + [THINK] + x[:3] + [END_THINK] + off + x[5:9], drafts=x[9:12], budget=S)], []),
        ("cold scan across 1024-token blocks", [dict(slot=1, committed=think + [x[i % 40] for i in range(2600)], drafts=[W, H], budget=S)], [0, 1, 2]),
        ("no budget: marker cache not kept, never penalized", [dict(slot=3, committed=think + x[:9], drafts=x[9:12], budget=-1)], []),
        ("budget exhausted + penalty", [dict(slot=0, committed=think + x[:6], drafts=[x[6], A], budget=4)], [0, 1, 2]),
        ("mixed batch", [dict(slot=0, committed=think + x[:9], drafts=[x[9]], budget=S),
                         dict(slot=1, committed=off + x[:6], drafts=[x[6], x[7]], budget=S),
                         dict(slot=2, committed=think + x[:3], drafts=[END_THINK, x[3]], budget=S)], [0, 1, 5]),
    ]


def test_named_cases(penalty=3.0):
    g = torch.Generator().manual_seed(0)
    for name, batch, want_rows in named_cases():
        slots = Slots(4, 4096)
        for req in batch:
            slots.add(req["slot"], req["committed"], req["budget"])
        n_rows = sum(len(r["drafts"]) + 1 for r in batch)
        logits = torch.randn((n_rows, VOCAB), generator=g, dtype=torch.float32)
        got = run(slots, batch, logits, [THINK], [END_THINK], [END_THINK], DEFAULT_IDS, penalty)
        want = reference(logits, batch, [THINK], [END_THINK], [END_THINK], DEFAULT_IDS, penalty)
        check("named cases match the reference", torch.equal(got, want),
              f"{name}: rows differ {rows_changed(want, got)}")
        penalized = [r for r in range(n_rows) if got[r, DEFAULT_IDS[0]] < logits[r, DEFAULT_IDS[0]] - 1]
        check("named cases penalize exactly the expected rows", penalized == want_rows,
              f"{name}: penalized {penalized}, expected {want_rows}")
        others = torch.ones(VOCAB, dtype=torch.bool)
        others[DEFAULT_IDS] = False
        others[END_THINK] = False
        check("nothing but the penalty ids (and a forced end) changes", torch.equal(got[:, others], logits[:, others]), name)
        if name.startswith("budget exhausted"):
            check("a real budget still forces </think>", bool((got[:, END_THINK] == 1.0e9).any()), name)


def test_stale_cache_without_budget(penalty=3.0):
    """A slot whose marker cache still says "inside reasoning" but whose request has no budget
    (only possible with the penalty off) must not be penalized: the cache is only kept for
    requests with a budget, so the kernel may not read it for the others."""
    x = [1000 + i for i in range(40)]
    think = [151644, 77091, 198, THINK, 198] + x[:12]
    off = [151644, 77091, 198, THINK, 271, END_THINK, 271] + x[:30]
    slots = Slots(2, 256)
    slots.add(1, think, _INT32_MAX)
    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    run(slots, [dict(slot=1, committed=think, drafts=[], budget=_INT32_MAX)], logits,
        [THINK], [END_THINK], [END_THINK], DEFAULT_IDS, penalty)          # the cache now holds a start
    slots.budget[1] = -1                                                 # add_request(no budget): no reset
    slots.commit(1, off)
    got = run(slots, [dict(slot=1, committed=off, drafts=[x[31]], budget=-1)], torch.zeros((2, VOCAB)),
              [THINK], [END_THINK], [END_THINK], DEFAULT_IDS, penalty)
    check("a stale cache is never read for a request without a budget", bool((got == 0).all()),
          got[:, DEFAULT_IDS])


def test_penalty_off_bitwise():
    """With the penalty off no request has a sentinel budget, so the kernels never touch the logits."""
    g = torch.Generator().manual_seed(1)
    for name, batch, _ in named_cases():
        slots = Slots(4, 4096)
        for req in batch:
            slots.add(req["slot"], req["committed"], -1 if req["budget"] == _INT32_MAX else req["budget"])
        n_rows = sum(len(r["drafts"]) + 1 for r in batch)
        logits = torch.randn((n_rows, VOCAB), generator=g, dtype=torch.float32)
        got = run(slots, batch, logits, [THINK], [END_THINK], [END_THINK], DEFAULT_IDS, 0.0)
        if all(r["budget"] in (-1, _INT32_MAX) for r in batch):
            check("penalty off: logits bitwise unchanged", torch.equal(got, logits), name)


def test_sentinel_never_forces():
    """The unreachable budget alone (penalty kernel not run) leaves every row untouched."""
    g = torch.Generator().manual_seed(2)
    for name, batch, _ in named_cases():
        if any(r["budget"] not in (-1, _INT32_MAX) for r in batch):
            continue
        slots = Slots(4, 4096)
        for req in batch:
            slots.add(req["slot"], req["committed"], _INT32_MAX)
        n_rows = sum(len(r["drafts"]) + 1 for r in batch)
        logits = torch.randn((n_rows, VOCAB), generator=g, dtype=torch.float32)
        got = run(slots, batch, logits, [THINK], [END_THINK], [END_THINK], DEFAULT_IDS, 0.0)
        check("the sentinel budget never forces an end", torch.equal(got, logits), name)


# -- random scenarios over a small vocabulary, several decode steps -----------------------------

def test_random(n_scenarios, seed=0):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    small_vocab = 16
    for sc in range(n_scenarios):
        # markers of one or two tokens (Qwen3.8's are one; the kernels take any length)
        start = [3] if rng.random() < 0.8 else [3, 5]
        natural_end = [4] if rng.random() < 0.8 else [4, 6]
        forced_end = natural_end if rng.random() < 0.7 else [9] + natural_end
        penalty_ids = sorted(rng.sample([7, 8, 10, 11, 12], rng.randint(1, 3)))
        penalty = rng.choice([0.5, 1.5, 3.0, 6.0, 100.0])
        p_marker = rng.choice([0.002, 0.02, 0.1])
        # 3 and 4 open the markers and never occur on their own; 5, 6 (the second halves of
        # the two-token markers) and 9 (the forced end's extra first token) do, as edge cases.
        ordinary = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        def tokens(n):
            """n tokens, markers whole (a marker cut by the end is an edge case too)."""
            out = []
            while len(out) < n:
                r = rng.random()
                out += start if r < p_marker else natural_end if r < 2 * p_marker else [rng.choice(ordinary)]
            return out[:n]

        slots = Slots(4, 4200)
        live = {}
        for slot in rng.sample(range(4), rng.randint(1, 4)):
            n = rng.choice([1, 2, 5, 30, 300, 1100, 2500])
            committed = tokens(n)
            if rng.random() < 0.5 and n >= len(start):
                at = n - rng.randint(len(start), max(len(start), min(n, 40)))
                committed[at:at + len(start)] = start
            budget = rng.choices([-1, _INT32_MAX, rng.randint(0, 60)], [1, 6, 3])[0]
            slots.add(slot, committed, budget)
            live[slot] = dict(slot=slot, committed=committed, budget=budget)
        for step in range(rng.randint(1, 3)):
            batch = []
            for req in live.values():
                req["drafts"] = tokens(rng.choice([0, 1, 3, 7]))
                batch.append(req)
            n_rows = sum(len(r["drafts"]) + 1 for r in batch)
            logits = torch.randn((n_rows, small_vocab), generator=g, dtype=torch.float32)
            got = run(slots, batch, logits, start, natural_end, forced_end, penalty_ids, penalty)
            want = reference(logits, batch, start, natural_end, forced_end, penalty_ids, penalty)
            check("random scenarios match the reference", torch.equal(got, want),
                  f"scenario {sc} step {step}: rows {rows_changed(want, got)} "
                  f"start={start} end={natural_end} forced={forced_end}")
            for req in batch:   # accept a prefix of the drafts plus one sampled token, then commit
                accepted = req["drafts"][:rng.randint(0, len(req["drafts"]))] + tokens(1)
                req["committed"] = req["committed"] + accepted
                slots.commit(req["slot"], req["committed"])


# -- config and token ids ------------------------------------------------------------------------

def test_config(model_dir):
    ok = ReasoningConfig()
    check("config: off by default", ok.think_penalty == 0 and ok.think_penalty_words == []
          and ok.think_penalty_token_ids is None)
    ReasoningConfig(think_penalty=100, think_penalty_words=["Wait"])
    for bad in (dict(think_penalty=3.0), dict(think_penalty=-1, think_penalty_words=["Wait"]),
                dict(think_penalty=100.5, think_penalty_words=["Wait"]),
                dict(think_penalty=float("nan"), think_penalty_words=["Wait"]),
                dict(think_penalty=float("inf"), think_penalty_words=["Wait"]), dict(think_penaltyy=3)):
        try:
            ReasoningConfig(**bad)
            check("config: bad values refused", False, f"{bad} was accepted")
        except Exception:   # noqa: BLE001 - pydantic wraps some as ValidationError
            check("config: bad values refused", True)

    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--model", str(model_dir), "--reasoning-parser", "qwen3", "--reasoning-config",
                              '{"think_penalty":3,"think_penalty_words":["Wait","Hmm","Alternatively"]}'])
    rc = EngineArgs.from_cli_args(args).reasoning_config
    check("config: the launcher's --reasoning-config parses", rc.think_penalty == 3.0
          and rc.think_penalty_words == DEFAULT_WORDS, rc)

    from vllm.tokenizers import get_tokenizer
    tok = get_tokenizer(str(model_dir))
    ids, kept, dropped = resolve_think_penalty_token_ids(tok, DEFAULT_WORDS, [THINK, END_THINK])
    check("ids: the default words give the six known ids", ids == DEFAULT_IDS, f"{ids} {kept}")
    ids, kept, dropped = resolve_think_penalty_token_ids(tok, ["hmm", "Double-check"], [THINK, END_THINK])
    check("ids: multi-token forms are dropped", ids == [84485] and kept == {" hmm": 84485}
          and len(dropped) == 3, f"{ids} {kept} {dropped}")
    ids, _, dropped = resolve_think_penalty_token_ids(tok, ["<think>", "<|im_end|>"], [THINK, END_THINK])
    check("ids: markers and special tokens are never penalized", ids == [] and len(dropped) == 4, f"{ids} {dropped}")

    # initialize_token_ids end to end: the qwen3 parser's markers, then the penalty ids.
    import vllm.config.reasoning as reasoning_mod
    real = reasoning_mod.cached_tokenizer_from_config
    reasoning_mod.cached_tokenizer_from_config = lambda model_config: tok
    try:
        rc = ReasoningConfig(reasoning_parser="qwen3", think_penalty=3.0, think_penalty_words=DEFAULT_WORDS)
        rc.initialize_token_ids(None)
        check("init: markers and penalty ids resolved", rc.enabled and rc.reasoning_start_token_ids == [THINK]
              and rc.natural_reasoning_end_token_ids == [END_THINK] and rc.think_penalty_token_ids == DEFAULT_IDS,
              rc.think_penalty_token_ids)
        rc = ReasoningConfig(reasoning_parser="qwen3", think_penalty=3.0, think_penalty_words=["Double-check"])
        try:
            rc.initialize_token_ids(None)
            check("init: nothing to penalize is refused", False, "accepted")
        except ValueError:
            check("init: nothing to penalize is refused", True)
        rc = ReasoningConfig(think_penalty=3.0, think_penalty_words=DEFAULT_WORDS)   # no parser, no markers
        try:
            rc.initialize_token_ids(None)
            check("init: a penalty without markers is refused", False, "accepted")
        except ValueError:
            check("init: a penalty without markers is refused", True)
        rc = ReasoningConfig(reasoning_parser="qwen3")
        rc.initialize_token_ids(None)
        check("init: penalty off resolves nothing extra", rc.enabled and rc.think_penalty_token_ids is None)
    finally:
        reasoning_mod.cached_tokenizer_from_config = real


# -- GPU: ThinkingBudgetState itself ---------------------------------------------------------------

def test_state_gpu(tok):
    from types import SimpleNamespace

    import vllm.config.reasoning as reasoning_mod
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

    real = reasoning_mod.cached_tokenizer_from_config
    reasoning_mod.cached_tokenizer_from_config = lambda model_config: tok
    try:
        def config(penalty):
            words = DEFAULT_WORDS if penalty else []
            rc = ReasoningConfig(reasoning_parser="qwen3", think_penalty=penalty, think_penalty_words=words)
            rc.initialize_token_ids(None)
            return rc
        cfg_on, cfg_off = config(3.0), config(0.0)
    finally:
        reasoning_mod.cached_tokenizer_from_config = real

    n, max_len = 4, 70000
    req_states = SimpleNamespace(
        max_num_reqs=n, device=torch.device("cuda"),
        all_token_ids=SimpleNamespace(gpu=torch.zeros((n, max_len), dtype=torch.int32, device="cuda")),
        total_len=SimpleNamespace(gpu=torch.zeros(n, dtype=torch.int32, device="cuda")))

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logging.getLogger("vllm.v1.worker.gpu.sample.thinking_budget").addHandler(handler)
    state = ThinkingBudgetState(req_states, cfg_on)
    check("state: the log line", any("Think penalty: -3.00 logits on 6 token ids" in m for m in records), records)

    x = [1000 + i for i in range(40)]
    def load(slot, committed):
        req_states.all_token_ids.gpu[slot].zero_()
        req_states.all_token_ids.gpu[slot, :len(committed)] = torch.tensor(committed, dtype=torch.int32)
        req_states.total_len.gpu[slot] = len(committed)

    def step(st, slot, drafts, penalty_ids=DEFAULT_IDS):
        committed_last = int(req_states.all_token_ids.gpu[slot, int(req_states.total_len.gpu[slot]) - 1])
        rows = len(drafts) + 1
        logits = torch.zeros((rows, VOCAB), dtype=torch.float32, device="cuda")
        t = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cuda")  # noqa: E731
        import numpy as np
        st.apply(logits, t([slot] * rows), t([slot]), np.array([slot]), t([committed_last] + drafts), t(list(range(rows))))
        return logits.cpu()

    # A long request in slot 1, deep inside its reasoning: every row penalized.
    think_long = [151644, 77091, 198, THINK, 198] + [x[i % 40] for i in range(30000)]
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    load(1, think_long)
    out = step(state, 1, [x[1], x[2], x[3]])
    check("state: a request without a budget is penalized inside reasoning",
          bool((out[:, DEFAULT_IDS] == -3.0).all()), out[:, DEFAULT_IDS])
    # The same slot reused by a shorter request with thinking off: nothing penalized.
    off = [151644, 77091, 198, THINK, 271, END_THINK, 271] + x[:10]
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    load(1, off)
    out = step(state, 1, [x[11], x[12]])
    check("state: a reused slot (shorter request) starts clean", bool((out == 0).all()), out[:, DEFAULT_IDS])
    # ... and by a longer one: the old scan position is inside it, so only the reset keeps it clean.
    off_long = [151644, 77091, 198, THINK, 271, END_THINK, 271] + [x[i % 40] for i in range(40000)]
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    load(1, think_long)
    step(state, 1, [])                          # warms the cache on the thinking request
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    load(1, off_long)
    out = step(state, 1, [x[3]])
    check("state: a reused slot (longer request) starts clean", bool((out == 0).all()), out[:, DEFAULT_IDS])

    # A real budget still forces </think>, with the penalty on top.
    state.add_request(2, SamplingParams(thinking_token_budget=4))
    state.apply_staged_writes()
    load(2, [151644, 77091, 198, THINK, 198] + x[:6])
    out = step(state, 2, [x[7]])
    check("state: a real budget forces </think> and the penalty still applies",
          bool((out[:, END_THINK] == 1.0e9).all()) and bool((out[:, DEFAULT_IDS] == -3.0).all()), out[:, END_THINK])

    # Penalty 0: no sentinel, no kernels, logits bitwise unchanged.
    state0 = ThinkingBudgetState(req_states, cfg_off)
    check("state: penalty 0 is off", state0.enabled and not state0.penalty_on)
    state0.add_request(1, SamplingParams())
    state0.apply_staged_writes()
    load(1, think_long)
    g = torch.Generator(device="cuda").manual_seed(3)
    logits = torch.randn((3, VOCAB), generator=g, dtype=torch.float32, device="cuda")
    before = logits.clone()
    t = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cuda")  # noqa: E731
    import numpy as np
    state0.apply(logits, t([1, 1, 1]), t([1]), np.array([1]), t([x[0], x[1], x[2]]), t([0, 1, 2]))
    check("state: penalty 0 leaves logits bitwise unchanged", torch.equal(logits, before))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--model", type=Path, help="model dir for the tokenizer "
                    "(default: models/Qwen3.8-27B-W4A16-AutoRound-fast, else the base dir)")
    ap.add_argument("--random", type=int, default=300, help="random scenarios (default 300)")
    args = ap.parse_args()
    model = args.model
    if model is None:
        for d in ("Qwen3.8-27B-W4A16-AutoRound-fast", "Qwen3.8-27B-W4A16-AutoRound"):
            if (REPO / "models" / d / "tokenizer.json").is_file():
                model = REPO / "models" / d
                break
    if model is None:
        print("no model dir with a tokenizer (prepare the model, or pass --model DIR)")
        return 2

    if MODE == "--cpu":
        test_config(model)
    test_named_cases()
    test_stale_cache_without_budget()
    test_penalty_off_bitwise()
    test_sentinel_never_forces()
    test_random(args.random)
    if MODE == "--gpu":
        from vllm.tokenizers import get_tokenizer
        test_state_gpu(get_tokenizer(str(model)))

    for name, (n_ok, n) in counts.items():
        print(f"{name:62s} {n_ok:4d}/{n:<4d} {'OK' if n_ok == n else 'FAIL'}")
    print(f"think penalty ({MODE[2:]}, {'interpreted' if MODE == '--cpu' else 'compiled'} kernels): "
          f"{sum(n for _, n in counts.values())} checks, {'OK' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
