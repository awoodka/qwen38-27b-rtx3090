#!/bin/bash
# The thinking-levers live session: every GPU-side check docs/thinking-levers.md reports, one
# server at a time, unattended (about 2 hours on an RTX 3090). Results go to
# bench/results/levers-<UTC time>/ (gitignored); summary.md is the digest.
#
# The GPU must be free for the whole run: run it under whatever pauses your other GPU work.
# Here that is llm-lab, in tmux:
#
#   tmux new -s levers
#   cd ~/llm-lab/lab && uv run lab gpu run --label fork/levers -- \
#     env BASELINE_REPO=$HOME/qwen-serving-bae2023 bash $HOME/qwen-serving-<c7>/bench/thinking_levers_session.sh
#
# BASELINE_REPO is a checkout of bae2023 (the commit this fork branches from) with its own venv.
# Step 1 measures how reproducible greedy output is across a restart there, which is what the
# levers-off parity in step 2 is judged against.
#
#  1. bae2023, booted twice: greedy outputs, then the same again after a restart.
#  2. This fork with the levers off: greedy parity, run_benchmarks.sh single twice (the second
#     counts), marker rates, the thinking_token_budget check, the render check.
#  3. REASONING_EFFORT=focused: the render check (tokenize = billed, focused text, every
#     top-level effort 200), marker rates.
#  4. THINK_PENALTY 1.5, 3, 6 and 100: marker rates and the content check; at 3 also the budget
#     check and run_benchmarks.sh twice, which the speed criterion (3% decode, 0.1 tokens/step)
#     compares with step 2.
#  5. THINK_PENALTY=3 with Actually, then with Maybe, added to the words: a screen for each.
#  6. bench/thinking_levers.py report.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK="$(dirname "$HERE")"
BASELINE_REPO=${BASELINE_REPO:?set BASELINE_REPO to a bae2023 checkout with its venv}
OUT=${OUT:-$HERE/results/levers-$(date -u +%Y%m%dT%H%M%S)}
HOST=127.0.0.1
PORT=18020
mkdir -p "$OUT"
CLIENT=(python3 "$HERE/thinking_levers.py" --out-dir "$OUT" --host "$HOST" --port "$PORT")
SERVER=""
FAILED=()

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
gpu_free() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

stop_server() {
  [ -n "$SERVER" ] || return 0
  kill -TERM "$SERVER" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER" 2>/dev/null || break; sleep 2; done
  if kill -0 "$SERVER" 2>/dev/null; then say "server $SERVER ignored SIGTERM for 120 s: SIGKILL"; kill -KILL "$SERVER"; fi
  wait "$SERVER" 2>/dev/null
  SERVER=""
  # The engine core is a child process that can outlive the API server by a few seconds.
  for _ in $(seq 30); do gpu_free && return 0; sleep 2; done
  say "the GPU is still busy after the server exited; killing what is left on it"
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -KILL "$pid" 2>/dev/null; done
  sleep 5
  gpu_free || { say "the GPU is still busy: giving up"; exit 1; }
}
trap 'stop_server' EXIT
trap 'say "interrupted"; exit 130' INT TERM

# start_server NAME REPO [VAR=VALUE...]: the launcher's defaults plus SPEC=dflash2 PREFIX_CACHE=1,
# the configuration the levers are measured on. The subshell execs, so $! is vLLM itself.
start_server() {
  local name=$1 repo=$2; shift 2
  gpu_free || { say "the GPU is busy before $name: refusing to start"; exit 1; }
  local fi_base=()
  # A separate FlashInfer JIT cache for this fork's runtime, so booting it neither rebuilds
  # kernels in the shared ~/.cache nor slows the next boot of the baseline (or hosted) runtime.
  [ "$repo" = "$FORK" ] && fi_base=(FLASHINFER_WORKSPACE_BASE="$FORK/.lab")
  say "boot $name ($repo) $*"
  (cd "$repo" && exec env -u VLLM_API_KEY CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" HOST=$HOST PORT=$PORT \
     SPEC=dflash2 PREFIX_CACHE=1 "${fi_base[@]}" "$@" bash single-user/start_qwen.sh) > "$OUT/server-$name.log" 2>&1 &
  SERVER=$!
  local t0
  t0=$(date +%s)
  until curl -sf -o /dev/null "http://$HOST:$PORT/health"; do
    if ! kill -0 "$SERVER" 2>/dev/null; then
      say "$name died during boot:"; tail -20 "$OUT/server-$name.log" | tee -a "$OUT/session.log"; SERVER=""; exit 1
    fi
    if [ $(( $(date +%s) - t0 )) -gt 900 ]; then say "$name: no /health after 900 s"; stop_server; exit 1; fi
    sleep 5
  done
  say "$name healthy after $(( $(date +%s) - t0 )) s"
}

# check NAME COMMAND...: run one client step; a failure is recorded, and the session goes on.
check() {
  local name=$1; shift
  if "$@" 2>&1 | tee -a "$OUT/session.log"; [ "${PIPESTATUS[0]}" = 0 ]; then return 0; fi
  FAILED+=("$name"); say "FAILED: $name"
}

# log_has NAME DESCRIPTION PATTERN [absent]: a line the server log must (or must not) contain.
log_has() {
  local f="$OUT/server-$1.log" found=no result
  grep -qE -- "$3" "$f" && found=yes
  if [ "${4:-}" = absent ]; then [ $found = no ] && result=ok || result=FAIL; else [ $found = yes ] && result=ok || result=FAIL; fi
  echo "$1: $2: $result" >> "$OUT/logchecks.txt"
  say "log $1: $2: $result"
  [ $result = ok ] || FAILED+=("log $1: $2")
}

# bench TAG: run_benchmarks.sh single twice on the fork's venv; the second run counts.
bench() {
  for n in 1 2; do
    say "run_benchmarks.sh single ($1, run $n)"
    (cd "$FORK" && HOST=$HOST PORT=$PORT OUT="$OUT/bench-raw-$1-$n" bash bench/run_benchmarks.sh single) > "$OUT/bench-$1-$n.txt" 2>&1
    grep "^ROW" "$OUT/bench-$1-$n.txt" | tee -a "$OUT/session.log"
  done
}

say "session: fork $(git -C "$FORK" rev-parse --short HEAD) at $FORK, baseline $(git -C "$BASELINE_REPO" rev-parse --short HEAD) at $BASELINE_REPO, out $OUT"
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm,driver_version --format=csv,noheader | tee -a "$OUT/session.log"

# 1. How reproducible greedy output is across a restart, on the commit this fork branches from.
start_server bae2023-a "$BASELINE_REPO"
check "greedy bae2023-a" "${CLIENT[@]}" greedy --tag bae2023-a
stop_server
start_server bae2023-b "$BASELINE_REPO"
check "greedy bae2023-b" "${CLIENT[@]}" greedy --tag bae2023-b --compare bae2023-a
stop_server

# 2. The fork with both levers off.
start_server off "$FORK"
log_has off "no think penalty" "Think penalty|think_penalty [0-9]" absent
check "greedy off" "${CLIENT[@]}" greedy --tag off --compare bae2023-a bae2023-b
bench off
check "rates off" "${CLIENT[@]}" rates --tag off
check "budget off" "${CLIENT[@]}" budget --tag off
check "render off" "${CLIENT[@]}" render --tag off --expect xhigh --efforts recorded
stop_server

# 3. The reasoning prompt.
start_server focused "$FORK" REASONING_EFFORT=focused
log_has focused "serves the fork's template" "templates/qwen3.8-27b.jinja"
log_has focused "default effort focused" "'reasoning_effort': 'focused'"
check "render focused" "${CLIENT[@]}" render --tag focused --expect focused
check "rates focused" "${CLIENT[@]}" rates --tag focused
stop_server

# 4. The token penalty.
for L in 1.5 3 6 100; do
  start_server "p$L" "$FORK" THINK_PENALTY=$L
  log_has "p$L" "the penalty is on" "Think penalty: -$(printf %.2f "$L") logits on 6 token ids"
  check "rates p$L" "${CLIENT[@]}" rates --tag "p$L"
  check "content p$L" "${CLIENT[@]}" content --tag "p$L"
  if [ "$L" = 3 ]; then
    check "budget p3" "${CLIENT[@]}" budget --tag p3
    bench p3
  fi
  stop_server
done

# 5. Screens: one more word each, at THINK_PENALTY=3.
for W in Actually Maybe; do
  tag="p3-$(echo "$W" | tr '[:upper:]' '[:lower:]')"
  start_server "$tag" "$FORK" THINK_PENALTY=3 THINK_PENALTY_WORDS="Wait,Hmm,Alternatively,$W"
  log_has "$tag" "the penalty is on, 8 token ids" "Think penalty: -3.00 logits on 8 token ids"
  check "rates $tag" "${CLIENT[@]}" rates --tag "$tag"
  check "content $tag" "${CLIENT[@]}" content --tag "$tag"
  stop_server
done

# 6. The digest.
"${CLIENT[@]}" report > /dev/null
say "summary: $OUT/summary.md"
if [ ${#FAILED[@]} -gt 0 ]; then
  say "failed steps (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
say "all steps passed"
