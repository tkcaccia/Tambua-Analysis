#!/usr/bin/env bash
# One authorized Chiamaka GPU scope. Never auto-retry or advance after failure.
set -uo pipefail

study=/mnt/sata_ssd/Tambua-benchmark/HEMIT
runtime=/mnt/sata_ssd/Tambua-orion-2026-10-19-v1/.venv/bin/python
program="$study/audit-code/run_gpu_bundle.py"
runner="$study/cellphenotyper-code/run_gigatime_on_crop.py"
ids="$study/audit-code/hemit_test_no_overlap_no_empty_ids.txt"
audit="$study/FULL_INPUT_AUDIT.json"
output="$study/gpu-validation-292-v1"

if [[ -e "$output" ]]; then
  echo "Output already exists; refusing duplicate GPU bundle" >&2
  exit 2
fi

gpu_before=$(timeout 10s nvidia-smi --query-gpu=uuid --format=csv,noheader) || exit 3
contexts_before=$(timeout 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 3
if [[ $(printf '%s\n' "$gpu_before" | sed '/^$/d' | wc -l) -ne 1 || -n "$contexts_before" ]]; then
  echo "Single Chiamaka GPU not idle; refusing launch" >&2
  exit 3
fi

mkdir -p "$study/launcher-evidence"
printf 'scope=292 fixed HEMIT test pairs\nhost=%s\nstart_utc=%s\ngpu_uuid=%s\n' \
  "$(hostname)" "$(date -u +%FT%TZ)" "$gpu_before" \
  > "$study/launcher-evidence/LAUNCH.txt"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  setsid --wait timeout --signal=TERM --kill-after=20s 7200s "$runtime" "$program" \
  --root "$study/extracted" --runner "$runner" --ids "$ids" \
  --audit "$audit" --output "$output" \
  2>&1 | tee "$study/launcher-evidence/RUN.log"
run_status=${PIPESTATUS[0]}

# Bounded cleanup proof. A vanished launcher or empty queue alone is insufficient.
cleanup=unknown
for attempt in 1 2 3 4 5; do
  case_pid=""
  case_pgid=""
  if [[ -f "$output/BUNDLE_STARTED.json" ]]; then
    case_pid=$("$runtime" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
      "$output/BUNDLE_STARTED.json" 2>/dev/null) || case_pid=""
    case_pgid=$("$runtime" -c 'import json,sys; print(json.load(open(sys.argv[1]))["process_group_id"])' \
      "$output/BUNDLE_STARTED.json" 2>/dev/null) || case_pgid=""
  fi
  alive=0
  if [[ -n "$case_pid" && -e "/proc/$case_pid" ]]; then alive=1; fi
  group_alive=0
  if [[ -n "$case_pgid" ]]; then
    group_alive=$(ps -eo pgid= | awk -v target="$case_pgid" '$1 == target {n++} END {print n+0}')
  fi
  contexts=$(timeout 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  telemetry_status=$?
  if [[ $telemetry_status -ne 0 ]]; then
    cleanup=unknown
    break
  fi
  if [[ $alive -eq 0 && $group_alive -eq 0 && -z "$contexts" ]]; then
    cleanup=verified
    break
  fi
  sleep 2
done

printf 'run_exit=%s\ncleanup=%s\nend_utc=%s\n' \
  "$run_status" "$cleanup" "$(date -u +%FT%TZ)" \
  > "$study/launcher-evidence/OUTCOME.txt"
if [[ $run_status -ne 0 || "$cleanup" != verified || ! -f "$output/BUNDLE_COMPLETE.json" ]]; then
  echo "HEMIT GPU bundle failed or cleanup unverified; stop and preserve evidence" >&2
  exit 4
fi
echo "HEMIT GPU bundle complete; process exit and GPU context cleanup verified"
