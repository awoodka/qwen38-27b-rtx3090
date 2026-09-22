"""Client for bench/thinking_levers_session.sh: the GPU-side checks and measurements behind
docs/thinking-levers.md, against one running server (default 127.0.0.1:18020, served name
qwen3.8-27b, the launcher's defaults). Stdlib only. Every subcommand writes JSON to --out-dir,
and `report` turns all of it into summary.md and summary.json.

  greedy  --tag T [--compare T1 ...]  temperature-0 outputs, thinking on, one request at a time;
                                      optionally compared with earlier tags (identical or not)
  rates   --tag T                     bench/prompts_thinking.jsonl at the model's default sampling:
                                      reasoning length, finish reasons, marker rates in reasoning
                                      (by token, via /tokenize, and by word), tokens per step
  content --tag T                     the penalized words must still reach the answer
  budget  --tag T                     thinking_token_budget still caps the reasoning
  render  --tag T [--expect LEVEL]    /tokenize counts what the server bills, the rendered system
                                      turn carries LEVEL's instruction, every top-level effort is 200
  report                              summary.md and summary.json from everything in --out-dir
"""
import argparse
import concurrent.futures
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "qwen3.8-27b"
MARKERS = ["Wait", "Hmm", "Alternatively", "Actually", "Maybe"]
DEFAULT_SET = ["Wait", "Hmm", "Alternatively"]
INSTRUCTIONS = {   # the first words of each level's system instruction
    "focused": "Reasoning effort is set to xhigh. Correctness comes first",
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully",
    "low": "Reasoning effort is set to low.",
}
SENTENCE_START = re.compile(r"(?:^|(?<=[.!?:])\s+|\n+)\s*([A-Z][A-Za-z'-]*)")


class Server:
    def __init__(self, host, port):
        self.base = f"http://{host}:{port}"

    def post(self, path, payload, timeout=3600):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, {"raw": body[:500]}

    def chat(self, messages, **kw):
        return self.post("/v1/chat/completions", {"model": MODEL, "messages": messages, **kw})

    def tokens(self, text):
        status, r = self.post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})
        if status != 200:
            raise RuntimeError(f"/tokenize: HTTP {status} {r}")
        return r["tokens"]

    def spec_totals(self):
        """(drafts, accepted tokens) from /metrics; tokens per step = 1 + accepted/drafts."""
        with urllib.request.urlopen(self.base + "/metrics", timeout=30) as r:
            text = r.read().decode()
        def total(name):
            return sum(float(line.split()[-1]) for line in text.splitlines()
                       if line.startswith(name) and not line.startswith("#"))
        return total("vllm:spec_decode_num_drafts_total"), total("vllm:spec_decode_num_accepted_tokens_total")


def prompts():
    return [json.loads(line) for line in open(HERE / "prompts_thinking.jsonl") if line.strip()]


def message_fields(r):
    m = r["choices"][0]["message"]
    u = r.get("usage") or {}
    return {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "",
            "finish_reason": r["choices"][0].get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}


def save(out_dir, name, data):
    (out_dir / name).write_text(json.dumps(data, indent=1, ensure_ascii=False))


def load(out_dir, name):
    p = out_dir / name
    return json.loads(p.read_text()) if p.exists() else None


# -- subcommands ----------------------------------------------------------------------------------

def cmd_greedy(srv, args):
    outputs = []
    for p in prompts():
        status, r = srv.chat([{"role": "user", "content": p["prompt"]}], temperature=0, max_tokens=768)
        outputs.append({"id": p["id"], "status": status, **(message_fields(r) if status == 200 else {"error": r})})
    compare = {}
    for other in args.compare or []:
        prev = load(args.out_dir, f"greedy-{other}.json")
        if prev is None:
            compare[other] = {"error": "missing"}
            continue
        same, first_diff = 0, {}
        for a, b in zip(outputs, prev["outputs"]):
            ta, tb = a.get("reasoning", "") + "\x00" + a.get("content", ""), b.get("reasoning", "") + "\x00" + b.get("content", "")
            if ta == tb:
                same += 1
            else:
                first_diff[a["id"]] = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        compare[other] = {"identical": same, "total": len(outputs), "first_divergence_char": first_diff}
    save(args.out_dir, f"greedy-{args.tag}.json", {"tag": args.tag, "outputs": outputs, "compare": compare})
    print(f"greedy {args.tag}: {sum(o['status'] == 200 for o in outputs)}/{len(outputs)} answered; "
          + "; ".join(f"vs {k}: {v.get('identical')}/{v.get('total')} identical" for k, v in compare.items()))
    return 0 if all(o["status"] == 200 for o in outputs) else 1


def marker_ids(srv):
    ids = {}
    for w in MARKERS:
        ids[w] = [t[0] for t in (srv.tokens(f) for f in (w, " " + w)) if len(t) == 1]
    return ids


def cmd_rates(srv, args):
    ids = marker_ids(srv)
    # The same seed for the same (prompt, sample) in every configuration, so configurations
    # differ by their setting rather than by their random draws.
    jobs = [(p, s, 1000 + 10 * i + s) for i, p in enumerate(prompts()) for s in range(args.samples)]
    d0, a0 = srv.spec_totals()
    t0 = time.time()

    def one(job):
        p, s, seed = job
        t = time.time()
        status, r = srv.chat([{"role": "user", "content": p["prompt"]}], max_tokens=args.max_tokens, seed=seed)
        rec = {"id": p["id"], "sample": s, "status": status, "seconds": round(time.time() - t, 2)}
        rec.update(message_fields(r) if status == 200 else {"error": r})
        return rec

    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        records = list(pool.map(one, jobs))
    wall = time.time() - t0
    d1, a1 = srv.spec_totals()
    for rec in records:
        text = rec.get("reasoning", "")
        toks = srv.tokens(text) if text else []
        rec["reasoning_tokens_retokenized"] = len(toks)
        rec["marker_tokens"] = {w: sum(t in ids[w] for t in toks) for w in MARKERS}
        rec["words"] = len(text.split())
        rec["marker_words"] = {w: len(re.findall(rf"\b{w}\b", text)) for w in MARKERS}
        starts = SENTENCE_START.findall(text)
        rec["marker_sentence_starts"] = {w: starts.count(w) for w in MARKERS}
    ok = [r for r in records if r["status"] == 200]
    rt = [r["reasoning_tokens"] for r in ok if r.get("reasoning_tokens") is not None]
    toks_total = sum(r["reasoning_tokens_retokenized"] for r in ok) or 1
    words_total = sum(r["words"] for r in ok) or 1
    agg = {
        "tag": args.tag, "requests": len(records), "answered": len(ok), "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "wall_seconds": round(wall, 1),
        "finish_reasons": {k: sum(r.get("finish_reason") == k for r in ok) for k in sorted({r.get("finish_reason") for r in ok}, key=str)},
        "reasoning_tokens_mean": round(statistics.mean(rt), 1) if rt else None,
        "reasoning_tokens_median": statistics.median(rt) if rt else None,
        "completion_tokens_total": sum(r.get("completion_tokens") or 0 for r in ok),
        "throughput_tok_s": round(sum(r.get("completion_tokens") or 0 for r in ok) / wall, 1),
        "tokens_per_step": round(1 + (a1 - a0) / (d1 - d0), 3) if d1 > d0 else None,
        "marker_ids": ids,
        "markers_per_1k_reasoning_tokens": {w: round(1000 * sum(r["marker_tokens"][w] for r in ok) / toks_total, 3) for w in MARKERS},
        "markers_per_1k_words": {w: round(1000 * sum(r["marker_words"][w] for r in ok) / words_total, 3) for w in MARKERS},
        "sentence_starts_per_1k_words": {w: round(1000 * sum(r["marker_sentence_starts"][w] for r in ok) / words_total, 3) for w in MARKERS},
    }
    agg["default_set_per_1k_reasoning_tokens"] = round(sum(agg["markers_per_1k_reasoning_tokens"][w] for w in DEFAULT_SET), 3)
    agg["default_set_per_1k_words"] = round(sum(agg["markers_per_1k_words"][w] for w in DEFAULT_SET), 3)
    save(args.out_dir, f"rates-{args.tag}.json", agg)
    with open(args.out_dir / f"rates-{args.tag}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"rates {args.tag}: {len(ok)}/{len(records)} answered, reasoning tokens mean {agg['reasoning_tokens_mean']}, "
          f"median {agg['reasoning_tokens_median']}, finish {agg['finish_reasons']}, default-set markers "
          f"{agg['default_set_per_1k_reasoning_tokens']}/1k tokens, tokens/step {agg['tokens_per_step']}, {wall:.0f} s")
    return 0 if len(ok) == len(records) else 1


def cmd_content(srv, args):
    checks = [
        ("reply-exactly", "Reply with exactly this text and nothing else: Wait, Hmm, Alternatively, Actually, Maybe.",
         ["Wait, Hmm, Alternatively, Actually, Maybe"]),
        ("code-wait", "Write a short Python example that starts three asyncio tasks and waits for them with "
         "asyncio.wait, then a threading example in which a consumer thread calls Condition.wait(). Code only.",
         ["asyncio.wait", ".wait("]),
    ]
    results = []
    for name, prompt, needles in checks:
        status, r = srv.chat([{"role": "user", "content": prompt}], max_tokens=6000, seed=7)
        f = message_fields(r) if status == 200 else {"content": "", "error": r}
        results.append({"check": name, "status": status, "passed": status == 200 and all(n in f["content"] for n in needles),
                        "needles": needles, "content": f["content"][:600], "finish_reason": f.get("finish_reason"),
                        "reasoning_tokens": f.get("reasoning_tokens")})
    save(args.out_dir, f"content-{args.tag}.json", {"tag": args.tag, "checks": results})
    print(f"content {args.tag}: " + ", ".join(f"{c['check']} {'ok' if c['passed'] else 'FAIL'}" for c in results))
    return 0 if all(c["passed"] for c in results) else 1


def cmd_budget(srv, args):
    status, r = srv.chat([{"role": "user", "content": "How many prime numbers are there between 1 and 50?"}],
                         max_tokens=400, thinking_token_budget=32, chat_template_kwargs={"enable_thinking": True})
    f = message_fields(r) if status == 200 else {"error": r}
    passed = status == 200 and f.get("reasoning_tokens") is not None and 0 < f["reasoning_tokens"] <= 33
    save(args.out_dir, f"budget-{args.tag}.json", {"tag": args.tag, "status": status, "passed": passed,
                                                   "reasoning_tokens": f.get("reasoning_tokens"), "content": f.get("content", "")[:300]})
    print(f"budget {args.tag}: HTTP {status}, reasoning_tokens {f.get('reasoning_tokens')} (budget 32): {'ok' if passed else 'FAIL'}")
    return 0 if passed else 1


def cmd_render(srv, args):
    msg = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
    tool = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    out = {"tag": args.tag, "expect": args.expect, "counts": {}, "efforts": {}}
    passed = True
    for label, extra in (("plain", {}), ("tools", {"tools": [tool]})):
        s, t = srv.post("/tokenize", {"model": MODEL, "messages": msg, "add_generation_prompt": True, **extra})
        s2, d = srv.post("/detokenize", {"model": MODEL, "tokens": t.get("tokens", [])})
        s3, r = srv.chat(msg, max_tokens=8, **extra)
        billed = (r.get("usage") or {}).get("prompt_tokens")
        text = d.get("prompt", "")
        level = next((k for k, v in INSTRUCTIONS.items() if v in text), "none")
        ok = s == 200 and s3 == 200 and t.get("count") == billed and (args.expect is None or level == args.expect)
        passed &= ok
        out["counts"][label] = {"tokenize": t.get("count"), "billed": billed, "instruction": level, "ok": ok}
    # The model's own template raises on high/minimal/max (a 400): with it, the statuses are
    # recorded; with the fork's template every OpenAI effort must answer 200.
    for effort in ("high", "minimal", "max", "none", "low", "medium", "xhigh"):
        s, r = srv.chat(msg, max_tokens=8, reasoning_effort=effort)
        out["efforts"][effort] = {"status": s, "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens")}
        if args.efforts == "required":
            passed &= s == 200
    out["efforts_required"] = args.efforts == "required"
    out["passed"] = passed
    save(args.out_dir, f"render-{args.tag}.json", out)
    print(f"render {args.tag}: {out['counts']} efforts {[(e, v['status']) for e, v in out['efforts'].items()]}: "
          f"{'ok' if passed else 'FAIL'}")
    return 0 if passed else 1


# -- report ---------------------------------------------------------------------------------------

def bench_c1(out_dir, tag, run=2):
    """C1 default-sampling decode and tokens/step from run_benchmarks.sh's ROW lines."""
    p = out_dir / f"bench-{tag}-{run}.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("ROW cohort C1 real prompts T=default"):
            dec = re.search(r"decode\(C/meanTPOT\)=([\d.]+)", line)
            ts = re.search(r"tok/step=([\d.]+)", line)
            return {"decode_tok_s": float(dec.group(1)) if dec else None, "tokens_per_step": float(ts.group(1)) if ts else None}
    return None


def cmd_report(args):
    d = args.out_dir
    summary = {"greedy": {}, "rates": {}, "content": {}, "budget": {}, "render": {}, "bench": {}, "log_checks": None}
    for p in sorted(d.glob("greedy-*.json")):
        g = json.loads(p.read_text())
        summary["greedy"][g["tag"]] = g["compare"]
    for p in sorted(d.glob("rates-*.json")):
        r = json.loads(p.read_text())
        summary["rates"][r["tag"]] = r
    for kind in ("content", "budget", "render"):
        for p in sorted(d.glob(f"{kind}-*.json")):
            x = json.loads(p.read_text())
            summary[kind][x["tag"]] = x
    for tag in ("off", "p3"):
        summary["bench"][tag] = {n: bench_c1(d, tag, n) for n in (1, 2)}
    if (d / "logchecks.txt").exists():
        summary["log_checks"] = (d / "logchecks.txt").read_text().splitlines()

    lines = ["# Thinking-levers live session", ""]
    lines += ["## Greedy parity (temperature 0, thinking on, 768 tokens, one request at a time)", ""]
    for tag, comp in summary["greedy"].items():
        for other, c in comp.items():
            lines.append(f"- {tag} vs {other}: {c.get('identical')}/{c.get('total')} identical"
                         + (f"; first divergence (chars) {c.get('first_divergence_char')}" if c.get("first_divergence_char") else ""))
    lines += ["", "## Speed (run_benchmarks.sh single, C1 at default sampling, second run after the restart)", ""]
    off, p3 = (summary["bench"].get("off") or {}).get(2), (summary["bench"].get("p3") or {}).get(2)
    if off and p3 and off["decode_tok_s"] and p3["decode_tok_s"]:
        dd = 100 * (p3["decode_tok_s"] / off["decode_tok_s"] - 1)
        dts = (p3["tokens_per_step"] or 0) - (off["tokens_per_step"] or 0)
        summary["speed_check"] = {"decode_delta_pct": round(dd, 2), "tokens_per_step_delta": round(dts, 3),
                                  "passed": abs(dd) <= 3 and abs(dts) <= 0.1}
        lines.append(f"- levers off: {off['decode_tok_s']} tok/s, {off['tokens_per_step']} tokens/step")
        lines.append(f"- THINK_PENALTY=3: {p3['decode_tok_s']} tok/s, {p3['tokens_per_step']} tokens/step")
        lines.append(f"- delta: {dd:+.1f}% decode, {dts:+.2f} tokens/step: "
                     f"{'within' if summary['speed_check']['passed'] else 'OUTSIDE'} 3% / 0.1")
    lines += ["", "## Reasoning and markers (bench/prompts_thinking.jsonl, default sampling)", "",
              "| config | answered | reasoning tokens mean / median | mean vs off | finish | default-set markers per 1k reasoning tokens | "
              + " | ".join(f"{w} /1k tok" for w in MARKERS) + " | default set /1k words | tokens/step | tok/s (batch) |",
              "|---|---|---|---|---|---|" + "---|" * len(MARKERS) + "---|---|---|"]
    order = ["off", "focused", "p1.5", "p3", "p6", "p100", "p3-actually", "p3-maybe"]
    base_mean = (summary["rates"].get("off") or {}).get("reasoning_tokens_mean")
    for tag in sorted(summary["rates"], key=lambda t: (order.index(t) if t in order else len(order), t)):
        r = summary["rates"][tag]
        vs = f"{100 * (r['reasoning_tokens_mean'] / base_mean - 1):+.1f}%" if base_mean and r["reasoning_tokens_mean"] else "-"
        lines.append(f"| {tag} | {r['answered']}/{r['requests']} | {r['reasoning_tokens_mean']} / {r['reasoning_tokens_median']} | {vs} | "
                     f"{r['finish_reasons']} | {r['default_set_per_1k_reasoning_tokens']} | "
                     + " | ".join(str(r["markers_per_1k_reasoning_tokens"][w]) for w in MARKERS)
                     + f" | {r['default_set_per_1k_words']} | {r['tokens_per_step']} | {r['throughput_tok_s']} |")
    lines += ["", "## Checks", ""]
    for tag, c in summary["content"].items():
        lines.append(f"- content {tag}: " + ", ".join(f"{x['check']} {'ok' if x['passed'] else 'FAIL'}" for x in c["checks"]))
    for tag, b in summary["budget"].items():
        lines.append(f"- budget {tag}: HTTP {b['status']}, reasoning_tokens {b['reasoning_tokens']} of 32: {'ok' if b['passed'] else 'FAIL'}")
    for tag, r in summary["render"].items():
        statuses = ", ".join(f"{e} {v['status']}" for e, v in r["efforts"].items())
        lines.append(f"- render {tag}: {r['counts']}; efforts ({'must be 200' if r.get('efforts_required') else 'recorded'}): {statuses}")
    for line in summary["log_checks"] or []:
        lines.append(f"- log: {line}")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    save(d, "summary.json", summary)
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["greedy", "rates", "content", "budget", "render", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag")
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--expect", choices=list(INSTRUCTIONS) + ["none"])
    ap.add_argument("--efforts", choices=["required", "recorded"], default="required",
                    help="render: must every top-level effort answer 200 (the fork's template)?")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18020)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        return cmd_report(args)
    if not args.tag:
        ap.error(f"{args.command} needs --tag")
    srv = Server(args.host, args.port)
    return {"greedy": cmd_greedy, "rates": cmd_rates, "content": cmd_content, "budget": cmd_budget,
            "render": cmd_render}[args.command](srv, args)


if __name__ == "__main__":
    sys.exit(main())
