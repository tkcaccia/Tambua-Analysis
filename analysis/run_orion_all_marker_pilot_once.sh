#!/usr/bin/env bash
# Inert until one separately authorized Chiamaka GPU pilot is approved.
# Never run on UCT HPC. One CRC04 case; no retries or later GPU descendants.
set -Eeuo pipefail

root=/home/chiamaka/Tambua-orion-all-marker-pilot-2026-10-20-v1
study=/mnt/sata_ssd/Tambua-orion-all-patient-2026-10-19
input="$study/cpu-prep/CRC04"
previous="$study/gpu-output/CRC04"
code="$study/gpu-code"
validator=/home/chiamaka/Tambua-orion-all-marker-pilot-code-2026-10-20/validate_orion_all_marker_pilot.py
python_bin=/mnt/sata_ssd/Tambua-orion-2026-10-19-v1/.venv/bin/python
cpu_bundle=/home/chiamaka/Tambua-monusac-expanded-cpu-inference-2026-10-20
cpu_pidfile=/home/chiamaka/Tambua-monusac-expanded-cpu-controller.pid
gpu_uuid=GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5
expected_rgb_sha256=5629a814cc11550e71a2d160d0c41f7e631dc965ad054f6a47713c0c71514ad9
expected_runner_sha256=e8d0bc2e261d6e2eb6627798649584a1fe0ad5e5c18420caea00663a132560ff
owned_pid=''
owned_pgid=''
owned_starttick=''

gpu_query() {
  timeout 10s nvidia-smi -i "$gpu_uuid" \
    --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
}

group_members() {
  if [[ -n "$owned_pgid" ]]; then
    ps -eo pid=,pgid=,stat=,args= | awk -v pg="$owned_pgid" '$2==pg && $3 !~ /^Z/ {print}'
  fi
}

stop_owned() {
  local members
  members=$(group_members)
  if [[ -n "$members" ]]; then
    kill -TERM -- "-$owned_pgid" 2>/dev/null || true
    sleep 5
    members=$(group_members)
    if [[ -n "$members" ]]; then
      kill -KILL -- "-$owned_pgid" 2>/dev/null || true
    fi
  fi
}

verify_cleanup() {
  local attempt
  for attempt in $(seq 1 15); do
    group_members >"$root/owned-processes-after.txt" || return 1
    gpu_query >"$root/gpu-after.txt" || return 1
    if [[ ! -s "$root/owned-processes-after.txt" && ! -s "$root/gpu-after.txt" ]]; then
      printf '%s\n' 'Owned group and assigned GPU context absent' >"$root/cleanup-verified.txt"
      return 0
    fi
    sleep 2
  done
  printf '%s\n' 'Cleanup unknown: stop and escalate; never restart this pilot' >"$root/cleanup-unknown.txt"
  return 1
}

on_error() {
  local status=$?
  trap - ERR INT TERM
  set +e
  stop_owned
  verify_cleanup
  printf '{"status":"failed","exit_code":%s,"cleanup_verified":%s}\n' \
    "$status" "$([[ -f "$root/cleanup-verified.txt" ]] && echo true || echo false)" >"$root/PILOT_FAILED.json"
  exit "${status:-1}"
}
on_interrupt() {
  trap - ERR INT TERM
  set +e
  stop_owned
  verify_cleanup
  printf '{"status":"interrupted","cleanup_verified":%s}\n' \
    "$([[ -f "$root/cleanup-verified.txt" ]] && echo true || echo false)" >"$root/PILOT_FAILED.json"
  exit 130
}
trap on_error ERR
trap on_interrupt INT TERM

if [[ -e "$root" ]]; then
  echo "Pilot path exists; no retry or overwrite: $root" >&2
  exit 65
fi
mkdir "$root"
mkdir "$root/CRC04"
sha256sum "$0" "$validator" >"$root/controller-code-sha256.txt"
hostname >"$root/host.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$root/started-utc.txt"

# The previously approved CPU bundle must be terminal and its exact controller
# no longer live. This check never stops or restarts that bundle.
if [[ ! -f "$cpu_bundle/BUNDLE_COMPLETE.json" && ! -f "$cpu_bundle/BUNDLE_FAILED.json" ]]; then
  echo 'MoNuSAC CPU bundle is not terminal' >&2
  false
fi
if [[ -f "$cpu_pidfile" ]]; then
  read -r cpu_pid cpu_starttick _ <"$cpu_pidfile"
  if [[ -r "/proc/$cpu_pid/stat" ]]; then
    current_tick=$(awk '{print $22}' "/proc/$cpu_pid/stat")
    current_state=$(awk '{print $3}' "/proc/$cpu_pid/stat")
    if [[ "$current_tick" == "$cpu_starttick" && "$current_state" != Z ]]; then
      echo 'MoNuSAC CPU controller still live' >&2
      false
    fi
  fi
fi

test -s "$input/PROVENANCE.json"
test -s "$input/COMPARTMENT_QC.json"
test -s "$validator"
test -s "$previous/CASE_COMPLETE.json"
printf '%s  %s\n' "$expected_rgb_sha256" "$input/he.ome.tif" | sha256sum -c - >"$root/input-hash-check.txt"
printf '%s  %s\n' "$expected_runner_sha256" "$code/run_gigatime_on_crop.py" | sha256sum -c - >"$root/runner-hash-check.txt"
(cd "$code" && sha256sum -c CODE_SHA256.txt) >"$root/frozen-code-hash-check.txt"
available_kib=$(df -Pk "$root" | awk 'NR==2 {print $4}')
if (( available_kib < 3145728 )); then
  echo 'Less than 3 GiB free for full-channel pilot' >&2
  false
fi
timeout 10s nvidia-smi -i "$gpu_uuid" --query-gpu=uuid,memory.total --format=csv,noheader >"$root/assigned-gpu.txt"
gpu_query >"$root/gpu-before.txt"
if [[ -s "$root/gpu-before.txt" ]]; then
  echo 'Assigned GPU already has a compute context' >&2
  false
fi

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu_uuid" OMP_NUM_THREADS=4
case_output="$root/CRC04"
mkdir "$case_output/gigatime" "$case_output/quantification"
set +e
setsid timeout --signal=TERM --kill-after=30s 600s \
  "$python_bin" -u "$code/run_gigatime_on_crop.py" \
  --image "$input/he.ome.tif" \
  --nuclei-mask "$input/nuclei_orion_ids.ome.tif" \
  --cyto-mask "$input/cytoplasm_orion_ids.ome.tif" \
  --cell-mask "$input/cells.ome.tif" \
  --quant-dir "$case_output/quantification" \
  --outdir "$case_output/gigatime" \
  --device cuda --repo-revision a6556030bdb8071857559e12b921620aa1bbac5b \
  --strict-target-mpp --target-mpp 0.25 --auto-parameters \
  --output-format zarr --jpg-markers DAPI,CK,CD68 \
  --seam-qc-mode warn --nuclear-qc-mode report --batch-size 2 \
  >"$case_output/inference.log" 2>&1 &
owned_pid=$!
owned_starttick=$(awk '{print $22}' "/proc/$owned_pid/stat")
owned_pgid=$(ps -o pgid= -p "$owned_pid" | tr -d ' ')
printf '%s %s %s\n' "$owned_pid" "$owned_starttick" "$owned_pgid" >"$root/owned-worker-identity.txt"
wait "$owned_pid"
result=$?
set -e
printf '%s\n' "$result" >"$root/worker-exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$root/worker-stopped-utc.txt"
if ! verify_cleanup; then
  echo 'Cleanup unknown after worker; stop and escalate' >&2
  false
fi
if (( result != 0 )); then
  echo "Pilot worker exited $result; no scorer or retry" >&2
  false
fi

export CUDA_VISIBLE_DEVICES=-1
"$python_bin" "$validator" --input-case "$input" --previous-case "$previous" \
  --pilot-case "$case_output" --output "$root/CRC04/TECHNICAL_COMPLETE.json" \
  >"$root/validation.log" 2>&1
printf '{"status":"technical_complete_pending_visual_review","patient":"CRC04"}\n' \
  >"$root/PILOT_COMPLETE.json.tmp"
mv "$root/PILOT_COMPLETE.json.tmp" "$root/PILOT_COMPLETE.json"
