#!/usr/bin/env bash
# One authorized, sequential nine-patient Chiamaka GPU bundle. Never retry.
# UCT HPC execution is forbidden.
set -Eeuo pipefail

root=/home/chiamaka/Tambua-orion-nine-raw-2026-10-20-v1
staged=/home/chiamaka/Tambua-orion-nine-raw-code-2026-10-20
study=/mnt/sata_ssd/Tambua-orion-all-patient-2026-10-19
python_bin=/mnt/sata_ssd/Tambua-orion-2026-10-19-v1/.venv/bin/python
gpu_uuid=GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5
runner="$study/gpu-code/run_gigatime_on_crop.py"
scope="$staged/orion-raw-nine-scope-2026-10-20.json"
validator="$staged/validate_orion_nine_raw_case.py"
patients=(CRC06 CRC08 CRC10 CRC12 CRC17 CRC19 CRC21 CRC23 CRC25)
deadline=$(( $(date +%s) + 7200 ))
owned_pid=''
owned_starttick=''
owned_pgid=''
active_case=''

gpu_query() {
  timeout 10s nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
}
group_members() {
  if [[ -n "$owned_pgid" ]]; then
    timeout 10s ps -eo pid=,pgid=,stat=,args= | awk -v pg="$owned_pgid" '$2==pg && $3 !~ /^Z/ {print}'
  fi
}
verify_cleanup() {
  local attempt
  rm -f "$root/cleanup-verified.txt"
  for attempt in $(seq 1 15); do
    if ! group_members >"$root/owned-processes-after.txt"; then
      printf 'Owned process inspection failed after %s\n' "$active_case" >"$root/CLEANUP_UNKNOWN.txt"
      return 1
    fi
    if ! gpu_query >"$root/gpu-after.txt"; then
      printf 'GPU context inspection failed after %s\n' "$active_case" >"$root/CLEANUP_UNKNOWN.txt"
      return 1
    fi
    if [[ ! -s "$root/owned-processes-after.txt" && ! -s "$root/gpu-after.txt" ]]; then
      printf 'Owned group and assigned GPU compute context absent after %s\n' "$active_case" >"$root/cleanup-verified.txt"
      return 0
    fi
    sleep 2
  done
  printf 'Cleanup unknown after %s; stop whole bundle\n' "$active_case" >"$root/CLEANUP_UNKNOWN.txt"
  return 1
}
stop_owned() {
  local members
  members=$(group_members) || return 1
  if [[ -n "$members" ]]; then
    kill -TERM -- "-$owned_pgid" 2>/dev/null || true
    sleep 5
    members=$(group_members) || return 1
    if [[ -n "$members" ]]; then kill -KILL -- "-$owned_pgid" 2>/dev/null || true; fi
  fi
}
on_failure() {
  local code=$?
  trap - ERR INT TERM
  set +e
  stop_owned
  verify_cleanup
  local cleaned=false
  [[ -f "$root/cleanup-verified.txt" && ! -f "$root/CLEANUP_UNKNOWN.txt" ]] && cleaned=true
  printf '{"status":"failed","case":"%s","exit_code":%s,"cleanup_verified":%s}\n' \
    "$active_case" "${code:-1}" "$cleaned" >"$root/BUNDLE_FAILED.json.tmp"
  mv "$root/BUNDLE_FAILED.json.tmp" "$root/BUNDLE_FAILED.json"
  exit "${code:-1}"
}
on_interrupt() {
  trap - ERR INT TERM
  set +e
  stop_owned
  verify_cleanup
  printf '{"status":"interrupted","case":"%s"}\n' "$active_case" >"$root/BUNDLE_FAILED.json.tmp"
  mv "$root/BUNDLE_FAILED.json.tmp" "$root/BUNDLE_FAILED.json"
  exit 130
}
trap on_failure ERR
trap on_interrupt INT TERM

if [[ -e "$root" ]]; then echo "Bundle already exists, refusing retry: $root" >&2; exit 65; fi
mkdir "$root"
hostname >"$root/host.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$root/started-utc.txt"
sha256sum "$0" "$scope" "$validator" >"$root/frozen-controller-code-sha256.txt"
printf '%s  %s\n' e8d0bc2e261d6e2eb6627798649584a1fe0ad5e5c18420caea00663a132560ff "$runner" | sha256sum -c - >"$root/runner-hash-check.txt"
(cd "$study/gpu-code" && sha256sum -c CODE_SHA256.txt) >"$root/frozen-inference-code-hash-check.txt"
if (( $(df -Pk "$root" | awk 'NR==2 {print $4}') < 6291456 )); then echo 'Under 6 GiB free' >&2; false; fi
timeout 10s nvidia-smi -i "$gpu_uuid" --query-gpu=uuid,memory.total --format=csv,noheader >"$root/assigned-gpu.txt"
gpu_query >"$root/gpu-before.txt"
if [[ -s "$root/gpu-before.txt" ]]; then echo 'Assigned GPU already has a compute context' >&2; false; fi

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu_uuid" OMP_NUM_THREADS=4
for patient in "${patients[@]}"; do
  active_case="$patient"
  if (( $(date +%s) + 720 > deadline )); then
    echo "Bundle admission deadline reached before $patient" >&2
    false
  fi
  input="$study/cpu-prep/$patient"
  previous="$study/gpu-output/$patient"
  current="$root/$patient"
  mkdir "$current"
  # Validate immutable input hashes and prior successful DAPI technical record
  # before touching the GPU for this patient.
  "$python_bin" - "$scope" "$patient" "$input" "$previous" <<'PY' >"$current/input-check.txt"
import hashlib,json,sys
from pathlib import Path
scope=json.loads(Path(sys.argv[1]).read_text())
patient=sys.argv[2]; source=Path(sys.argv[3]); prior=Path(sys.argv[4])
case=next(x for x in scope['cases'] if x['patient']==patient)
for name,want in case['input_sha256'].items():
    got=hashlib.sha256((source/name).read_bytes()).hexdigest()
    if got!=want:raise SystemExit(f'Input hash mismatch {patient}/{name}')
if hashlib.sha256((prior/'CASE_COMPLETE.json').read_bytes()).hexdigest()!=case['prior_complete_sha256']:
    raise SystemExit(f'Prior DAPI QC hash mismatch {patient}')
report=json.loads((prior/'CASE_COMPLETE.json').read_text())
if report.get('technical_qc_passed') is not True:raise SystemExit('Prior DAPI QC not passed')
print('Frozen input and prior technical QC verified:',patient)
PY
  gpu_query >"$current/gpu-before.txt"
  if [[ -s "$current/gpu-before.txt" ]]; then echo "GPU context before $patient" >&2; false; fi
  mkdir "$current/gigatime" "$current/quantification"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$current/started-utc.txt"
  set +e
  setsid timeout --signal=TERM --kill-after=30s 600s \
    "$python_bin" -u "$runner" \
    --image "$input/he.ome.tif" \
    --nuclei-mask "$input/nuclei_orion_ids.ome.tif" \
    --cyto-mask "$input/cytoplasm_orion_ids.ome.tif" \
    --cell-mask "$input/cells.ome.tif" \
    --quant-dir "$current/quantification" --outdir "$current/gigatime" \
    --device cuda --repo-revision a6556030bdb8071857559e12b921620aa1bbac5b \
    --strict-target-mpp --target-mpp 0.25 --auto-parameters \
    --output-format zarr --jpg-markers DAPI,CK,CD68 \
    --seam-qc-mode warn --nuclear-qc-mode report --batch-size 2 \
    >"$current/inference.log" 2>&1 &
  owned_pid=$!
  owned_starttick=$(awk '{print $22}' "/proc/$owned_pid/stat")
  owned_pgid=$(ps -o pgid= -p "$owned_pid" | tr -d ' ')
  controller_pgid=$(ps -o pgid= -p "$$" | tr -d ' ')
  if [[ -z "$owned_starttick" || -z "$owned_pgid" || "$owned_pgid" == "$controller_pgid" ]]; then
    set -e
    echo 'Could not record owned GPU process identity' >&2
    false
  fi
  printf '%s %s %s\n' "$owned_pid" "$owned_starttick" "$owned_pgid" >"$current/owned-worker-identity.txt"
  wait "$owned_pid"
  result=$?
  set -e
  printf '%s\n' "$result" >"$current/worker-exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$current/worker-stopped-utc.txt"
  verify_cleanup
  cp "$root/cleanup-verified.txt" "$current/cleanup-verified.txt"
  if (( result != 0 )); then echo "$patient worker failed $result; no later case" >&2; false; fi
  export CUDA_VISIBLE_DEVICES=-1
  "$python_bin" "$validator" --scope "$scope" --patient "$patient" \
    --input-case "$input" --previous-case "$previous" --current-case "$current" \
    --output "$current/TECHNICAL_COMPLETE.json" >"$current/validation.log" 2>&1
  export CUDA_VISIBLE_DEVICES="$gpu_uuid"
  printf '%s\n' "$patient" >>"$root/patients-completed.txt"
  owned_pid=''; owned_pgid=''; owned_starttick=''
done
"$python_bin" - "$root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);patients='CRC06 CRC08 CRC10 CRC12 CRC17 CRC19 CRC21 CRC23 CRC25'.split()
rows=[json.loads((root/p/'TECHNICAL_COMPLETE.json').read_text()) for p in patients]
if [x['patient'] for x in rows]!=patients:raise SystemExit('Incomplete ordered bundle')
temporary=root/'NINE_PATIENTS_COMPLETE.json.tmp'
temporary.write_text(json.dumps({'status':'technical_complete_pending_visual_and_protein_validation','cases':rows},indent=2)+'\n')
temporary.replace(root/'NINE_PATIENTS_COMPLETE.json')
PY
