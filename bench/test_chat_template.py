"""Render-matrix test for templates/qwen3.8-27b.jinja, the fork's chat template. CPU only.

The contract (README "What this fork changes"): for every input the model's own
chat_template.jinja accepts, the fork's template renders byte-for-byte the same string. On
top of that, reasoning_effort 'focused' renders its own instruction where xhigh renders
xhigh's, max/high render exactly as xhigh and minimal/none exactly as low (the original
raises on all four), and any other effort the original rejects still raises.

Rendering goes through the tokenizer's apply_chat_template, the call vLLM makes
(vllm/renderers/hf.py safe_apply_chat_template), so whitespace control and the sandbox are
the ones the server uses.

  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_chat_template.py [--model DIR]
"""
import argparse
import hashlib
import itertools
import sys
from pathlib import Path

import jinja2
import jinja2.ext
import jinja2.meta
import jinja2.sandbox
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
FORK_TEMPLATE = REPO / "templates" / "qwen3.8-27b.jinja"
# The copy this template was made from (dbirks/Qwen3.8-27B-W4A16-AutoRound, identical in
# the -fast dir that prepare/fetch_fast_variant.py builds).
SOURCE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"

XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
         "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
         "and clarity in the final answer.")
FOCUSED = ("Reasoning effort is set to xhigh. Correctness comes first, so think as carefully as the "
           "task requires, but make every step move the solution forward. Compare approaches "
           "briefly, choose one, and commit to it. Check each result once, when you derive it; once "
           "it checks out, treat it as settled. Do not re-verify settled steps, restart from "
           "scratch, or reopen a choice you have already made unless you find a specific error; if "
           "you find one, fix it where it occurred and continue from there. When you are confident "
           "in the answer, stop thinking and give the final answer.")

UNSET = object()   # the kwarg is not passed at all (the template sees it undefined)

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}

SINGLE = [{"role": "user", "content": "What is 17 * 23?"}]
MULTI = [
    {"role": "user", "content": "What is 17 * 23?"},
    {"role": "assistant", "content": "391.", "reasoning_content": "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391."},
    {"role": "user", "content": [{"type": "text", "text": "And the weather in Aarhus?"}]},
    {"role": "assistant", "content": "", "reasoning_content": "Use the tool.",
     "tool_calls": [{"type": "function", "function": {"name": "get_weather",
                                                      "arguments": {"city": "Aarhus", "unit": "c"}}}]},
    {"role": "tool", "content": "{\"temp\": 14, \"sky\": \"overcast\"}"},
    {"role": "assistant", "content": "14 C and overcast.", "reasoning_content": "Report it."},
    {"role": "user", "content": "Thanks. One word: warm or cold?"},
]
CONVERSATIONS = {"single": SINGLE, "multi": MULTI}
SYSTEMS = {"no-system": None, "system": "You are a careful assistant.", "empty-system": ""}
TOOLS = {"no-tools": None, "tools": [WEATHER_TOOL]}
THINKING = {"thinking-undef": UNSET, "thinking-on": True, "thinking-off": False}
PRESERVE = {"preserve-undef": UNSET, "preserve-on": True, "preserve-off": False}
GEN_PROMPT = {"gen": True, "no-gen": False}
# The original's own vocabulary, plus effort not passed at all.
EFFORTS = {"effort-undef": UNSET, "xhigh": "xhigh", "medium": "medium", "low": "low"}
ALIASES = {"max": "xhigh", "high": "xhigh", "minimal": "low", "none": "low"}
# Rejected by the original and by the fork alike (matching is exact and case-sensitive).
UNKNOWN = ["bogus", "XHIGH", "High", "Focused", "focus", "extra-high", "", None]


class Renderer:
    def __init__(self, tokenizer, template):
        self.tok, self.template = tokenizer, template

    def __call__(self, messages, tools, gen, **kwargs):
        kw = {k: v for k, v in kwargs.items() if v is not UNSET}
        try:
            return self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template,
                                                tokenize=False, add_generation_prompt=gen, **kw)
        except jinja2.exceptions.TemplateError as e:
            return e   # a raise_exception() from the template


def matrix():
    """Every combination of the axes above: 2 x 3 x 2 x 3 x 3 x 2 conversations x kwargs."""
    for (cn, conv), (sn, sysc), (tn, tools), (hn, think), (pn, keep), (gn, gen) in itertools.product(
            CONVERSATIONS.items(), SYSTEMS.items(), TOOLS.items(), THINKING.items(),
            PRESERVE.items(), GEN_PROMPT.items()):
        messages = ([{"role": "system", "content": sysc}] if sysc is not None else []) + conv
        yield f"{cn}/{sn}/{tn}/{hn}/{pn}/{gn}", messages, tools, gen, think, keep


def template_vars(text):
    """The undeclared variables vLLM derives from a template to filter chat_template_kwargs
    (vllm/renderers/hf.py _resolve_chat_template_kwargs, same environment settings)."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                        extensions=[jinja2.ext.loopcontrols])
    return jinja2.meta.find_undeclared_variables(env.parse(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="model dir holding chat_template.jinja and the "
                    "tokenizer (default: models/Qwen3.8-27B-W4A16-AutoRound-fast, else the base dir)")
    args = ap.parse_args()
    model = args.model
    if model is None:
        for d in ("Qwen3.8-27B-W4A16-AutoRound-fast", "Qwen3.8-27B-W4A16-AutoRound"):
            if (REPO / "models" / d / "chat_template.jinja").is_file():
                model = REPO / "models" / d
                break
    if model is None or not (model / "chat_template.jinja").is_file():
        print("no model dir with chat_template.jinja (prepare the model, or pass --model DIR)")
        return 2

    original_text = (model / "chat_template.jinja").read_text(encoding="utf-8")
    fork_text = FORK_TEMPLATE.read_text(encoding="utf-8")
    sha = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
    if sha != SOURCE_SHA256:
        print(f"NOTE: {model.name}/chat_template.jinja is sha256 {sha[:12]}..., not the {SOURCE_SHA256[:12]}... "
              f"the fork's template was copied from; comparing against yours anyway.")
    tok = AutoTokenizer.from_pretrained(model)
    orig, fork = Renderer(tok, original_text), Renderer(tok, fork_text)

    fails = 0
    checks = {}

    def check(name, ok, detail=""):
        nonlocal fails
        n_ok, n = checks.get(name, (0, 0))
        checks[name] = (n_ok + bool(ok), n + 1)
        if not ok:
            fails += 1
            if checks[name][1] - checks[name][0] <= 3:   # the first few of each kind
                print(f"FAIL {name}: {detail}")

    # 1. Byte-identical wherever the original accepts the input. With thinking off the
    #    original never looks at reasoning_effort, so any value is accepted there, and the
    #    fork must ignore it the same way.
    for case, messages, tools, gen, think, keep in matrix():
        efforts = dict(EFFORTS)
        if think is False:
            efforts.update({v: v for v in ("focused", *ALIASES, "bogus")})
        for en, effort in efforts.items():
            kw = dict(enable_thinking=think, preserve_thinking=keep, reasoning_effort=effort)
            want = orig(messages, tools, gen, **kw)
            if isinstance(want, Exception):
                check("original accepts the matrix", False, f"{case}/{en}: {want}")
                continue
            got = fork(messages, tools, gen, **kw)
            check("identical to the original", got == want, f"{case}/{en}")

    # 2. focused = the xhigh render with xhigh's instruction swapped for focused's; with
    #    thinking off it is ignored like every other effort (covered in 1).
    # 3. max/high = xhigh and minimal/none = low, byte for byte; the original raises on all four.
    for case, messages, tools, gen, think, keep in matrix():
        if think is False:
            continue
        base = dict(enable_thinking=think, preserve_thinking=keep)
        xhigh = orig(messages, tools, gen, reasoning_effort="xhigh", **base)
        got = fork(messages, tools, gen, reasoning_effort="focused", **base)
        check("focused renders its own instruction",
              isinstance(got, str) and got.count(FOCUSED) == 1 and XHIGH not in got
              and got == xhigh.replace(XHIGH, FOCUSED), case)
        for alias, level in ALIASES.items():
            want = orig(messages, tools, gen, reasoning_effort=level, **base)
            check("alias renders as its level", fork(messages, tools, gen, reasoning_effort=alias, **base) == want,
                  f"{case}/{alias}")
            check("original raises on the alias", isinstance(orig(messages, tools, gen, reasoning_effort=alias, **base),
                                                             Exception), f"{case}/{alias}")

    # 4. Anything else the original rejects, the fork rejects too, and says what it accepts.
    for effort in UNKNOWN:
        for think in (UNSET, True):
            for tools in (None, [WEATHER_TOOL]):
                kw = dict(enable_thinking=think, reasoning_effort=effort)
                o, f = orig(SINGLE, tools, True, **kw), fork(SINGLE, tools, True, **kw)
                check("unknown effort still raises", isinstance(o, Exception) and isinstance(f, Exception)
                      and "focused" in str(f) and "max and high mean xhigh" in str(f), f"{effort!r}: {f!r}"[:200])

    # 5. The rest of the original's refusals are untouched.
    invalid = {
        "system not first": [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}],
        "no user query": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
        "unknown role": [{"role": "user", "content": "hi"}, {"role": "narrator", "content": "x"}],
    }   # (an empty list never reaches the template: transformers refuses it first)
    for name, messages in invalid.items():
        o, f = orig(messages, None, True), fork(messages, None, True)
        check("invalid input raises in both", isinstance(o, Exception) and isinstance(f, Exception)
              and str(o) == str(f), f"{name}: {o!r} / {f!r}")

    # 6. vLLM passes a chat_template_kwarg through only if the template reads it: the set of
    #    variables it sees must not change.
    check("same template variables for vLLM's kwargs filter", template_vars(original_text) == template_vars(fork_text),
          f"{sorted(template_vars(original_text) ^ template_vars(fork_text))}")

    total = sum(n for _, n in checks.values())
    for name, (n_ok, n) in checks.items():
        print(f"{name:52s} {n_ok:4d}/{n:<4d} {'OK' if n_ok == n else 'FAIL'}")
    print(f"chat template ({FORK_TEMPLATE.relative_to(REPO)} vs {model.name}/chat_template.jinja): "
          f"{total} checks, {'OK' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
