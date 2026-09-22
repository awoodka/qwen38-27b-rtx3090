#!/bin/bash
# Argv test for the fork's knobs in single-user/start_qwen.sh. No GPU, no vLLM, no models.
#
# Runs the launcher inside a throwaway repo whose venv/bin/vllm is a stub that writes its argv,
# one argument per line, and whose models/ holds empty stand-ins. Checks that:
#   - with no fork knob set, the argv and the launcher's own output equal upstream's
#     (bae2023, the commit this fork branches from) byte for byte, over a spread of
#     SPEC/CTX/TOOLS/... settings, including the lab's hosted 64k-dflash2 config;
#   - each REASONING_EFFORT level adds exactly --chat-template <repo>/templates/qwen3.8-27b.jinja
#     --default-chat-template-kwargs {"reasoning_effort":"<level>"} right after
#     --reasoning-parser qwen3, and nothing else;
#   - anything else in REASONING_EFFORT exits 1 before vLLM starts.
#
#   bash bench/test_launcher_args.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=bae2023ffc98753d337d2d2041784a277599a4c4
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

mkdir -p "$T/single-user" "$T/venv/bin" "$T/templates" \
         "$T/models/Qwen3.8-27B-W4A16-AutoRound-fast" "$T/models/Qwen3.8-27B-DFlash2-W4A16"
: > "$T/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
cp "$REPO/templates/qwen3.8-27b.jinja" "$T/templates/"
printf '#!/bin/bash\nprintf "%%s\\n" "$@" > "$FAKE_VLLM_ARGV"\n' > "$T/venv/bin/vllm"
chmod +x "$T/venv/bin/vllm"
git -C "$REPO" show "$BASE:single-user/start_qwen.sh" > "$T/upstream.sh" 2>/dev/null || {
  echo "cannot read single-user/start_qwen.sh at $BASE (shallow clone? git fetch --unshallow)"; exit 2; }
cp "$REPO/single-user/start_qwen.sh" "$T/fork.sh"

FAILS=0
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }

# run NAME LAUNCHER VAR=VALUE... -> $T/NAME.argv (absent if vLLM never started), .out, .rc.
# The environment is exactly what is given, plus PATH/HOME, so nothing from the caller leaks in.
# VLLM_OFFLOAD_KEEP_SHM=1 keeps the launcher's stale-region sweep away from the real /dev/shm.
run() {
  local name=$1 launcher=$2; shift 2
  cp "$T/$launcher.sh" "$T/single-user/start_qwen.sh"
  rm -f "$T/$name.argv"
  env -i PATH="/usr/bin:/bin" HOME="$T" VLLM_OFFLOAD_KEEP_SHM=1 FAKE_VLLM_ARGV="$T/$name.argv" "$@" \
    bash "$T/single-user/start_qwen.sh" > "$T/$name.out" 2>&1
  echo $? > "$T/$name.rc"
}

# 1. No fork knob: identical to upstream. A config is shell words, eval'd so a quoted value
# (EXTRA_ARGS with spaces) stays one word. HOSTED is the full environment llm-lab's
# supervisor passes for its hosted 64k DFlash2 server, with the paths moved into $T.
HOSTED="HOST=127.0.0.1 PORT=8080 MODEL=$T/models/Qwen3.8-27B-W4A16-AutoRound-fast SPEC=dflash2 CTX=fast \
MAX_LEN=65536 PREFIX_CACHE=1 KV_MEM=5583457484 MAX_SEQS=8 GPU_UTIL=0.93 DFLASH_TOKENS=7 LOOKUP=1 TOOLS=1 \
VISION=0 DRAFT=$T/models/Qwen3.8-27B-DFlash2-W4A16 'EXTRA_ARGS=--served-model-name lab/64k-dflash2 --load-format auto'"
CONFIGS=(
  ""
  "SPEC=dflash2"
  "$HOSTED"
  "SPEC=dflash2 CTX=long"
  "SPEC=dflash2 CTX=huge PREFIX_CACHE=1"
  "SPEC=dflash2 DFLASH_TOKENS=15"
  "SPEC=off"
  "SPEC=mtp CTX=long"
  "TOOLS=0 REQ_METRICS=1 VISION=1"
  "SPEC=dflash2 REASONING_EFFORT="
)
n=0
for cfg in "${CONFIGS[@]}"; do
  n=$((n + 1))
  eval "run up$n upstream $cfg"
  eval "run fk$n fork $cfg"
  label="${cfg:0:40}"
  if [ ! -s "$T/up$n.argv" ]; then
    fail "[${label:-defaults}] upstream launcher did not reach vllm: $(tail -2 "$T/up$n.out")"
  elif ! cmp -s "$T/up$n.argv" "$T/fk$n.argv"; then
    fail "[${label:-defaults}] argv differs from upstream:"; diff "$T/up$n.argv" "$T/fk$n.argv" | head -10
  elif ! cmp -s "$T/up$n.out" "$T/fk$n.out" || ! cmp -s "$T/up$n.rc" "$T/fk$n.rc"; then
    fail "[${label:-defaults}] launcher output differs from upstream:"; diff "$T/up$n.out" "$T/fk$n.out" | head -10
  fi
done
echo "no knobs = upstream argv: ${#CONFIGS[@]} configs checked"

# 2. Each level adds exactly its four arguments after --reasoning-parser qwen3.
expect_with() {   # upstream argv file, extra args... -> the argv with the extras spliced in
  local file=$1; shift
  awk -v extra="$(printf '%s\n' "$@")" '{ print } prev == "--reasoning-parser" && $0 == "qwen3" { print extra } { prev = $0 }' "$file"
}
for base_cfg in "" "$HOSTED"; do
  eval "run up upstream $base_cfg"
  for level in xhigh focused medium low; do
    eval "run fk fork $base_cfg REASONING_EFFORT=$level"
    expect_with "$T/up.argv" --chat-template "$T/templates/qwen3.8-27b.jinja" \
      --default-chat-template-kwargs "{\"reasoning_effort\":\"$level\"}" > "$T/want.argv"
    if ! cmp -s "$T/want.argv" "$T/fk.argv"; then
      fail "[REASONING_EFFORT=$level ${base_cfg:0:40}] argv is not upstream + the four flags:"
      diff "$T/want.argv" "$T/fk.argv" | head -10
    fi
  done
done
# EXTRA_ARGS comes later, so an explicit --chat-template there still wins in vLLM's argparse.
run fk fork SPEC=dflash2 REASONING_EFFORT=focused EXTRA_ARGS="--chat-template /elsewhere.jinja"
order=$(awk 'prev == "--chat-template" { printf "%s ", $0 } { prev = $0 }' "$T/fk.argv")
[ "$order" = "$T/templates/qwen3.8-27b.jinja /elsewhere.jinja " ] \
  || fail "[EXTRA_ARGS --chat-template] should come after the knob's: got $order"
echo "REASONING_EFFORT levels: xhigh focused medium low checked"

# 3. Anything else refuses before vLLM starts.
for bad in bogus high max minimal none XHIGH Focused " focused" "focused " "xhigh,low"; do
  run bad fork SPEC=dflash2 "REASONING_EFFORT=$bad"
  if [ "$(cat "$T/bad.rc")" != 1 ] || [ -e "$T/bad.argv" ] || ! grep -q "is not a level" "$T/bad.out"; then
    fail "[REASONING_EFFORT='$bad'] expected exit 1 before vllm, got rc=$(cat "$T/bad.rc"): $(tail -1 "$T/bad.out")"
  fi
done
echo "bad REASONING_EFFORT values refused"

echo "launcher args: $([ $FAILS = 0 ] && echo OK || echo "$FAILS FAILURES")"
[ $FAILS = 0 ]
