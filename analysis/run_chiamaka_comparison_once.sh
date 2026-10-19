#!/usr/bin/env bash
# One authorized comparison bundle on Chiamaka; intentionally not restartable.
set -Eeuo pipefail

case_dir=/mnt/sata_ssd/Tambua-comparison-2026-10-19-v1
source_case=/mnt/sata_ssd/Tambua-orion-2026-10-19-v2
container=/mnt/sata_ssd/CellPhenotyper_releases/2.8/out/cellphenotyper-2.8-gpu-amd64.sif
gpu_uuid=GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5
cd "$case_dir"
mkdir evidence/launch_guard
sha256sum -c evidence/staged_sha256.txt > evidence/hash_check.log
printf '%s\n' "$gpu_uuid" > evidence/assigned_gpu_uuid.txt
date -u +%Y-%m-%dT%H:%M:%SZ > evidence/started_utc.txt
timeout 10s nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > evidence/gpu_before.csv
if [[ -s evidence/gpu_before.csv ]]; then
  echo 'Assigned GPU is busy before the bundle; stop.' > evidence/failure.txt
  exit 70
fi

active_pid=''
active_pgid=''
active_starttick=''
verify_cleanup() {
  local method="$1" step
  for step in $(seq 1 15); do
    ps -eo pid=,pgid=,args= | awk -v pg="$active_pgid" '$2 == pg {print}' > "evidence/${method}_process_after.txt"
    timeout 10s nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "evidence/${method}_gpu_after.csv" || return 1
    if [[ ! -s "evidence/${method}_process_after.txt" && ! -s "evidence/${method}_gpu_after.csv" ]]; then
      echo 'Verified no owned process group member or assigned GPU compute context' > "evidence/${method}_cleanup.txt"
      return 0
    fi
    sleep 2
  done
  echo 'Cleanup unknown or failed; bundle stopped' > "evidence/${method}_cleanup.txt"
  return 1
}
stop_owned() {
  local method="$1" current_tick
  if [[ -n "$active_pid" && -r "/proc/$active_pid/stat" ]]; then
    current_tick=$(awk '{print $22}' "/proc/$active_pid/stat")
    if [[ "$current_tick" == "$active_starttick" ]]; then
      kill -TERM -- "-$active_pgid" 2>/dev/null || true
      sleep 10
      if [[ -r "/proc/$active_pid/stat" ]]; then
        kill -KILL -- "-$active_pgid" 2>/dev/null || true
      fi
    fi
  fi
  verify_cleanup "$method" || true
}
on_interrupt() {
  trap - INT TERM
  echo 'Bundle interrupted; no further method will run' > evidence/failure.txt
  stop_owned interrupted
  exit 130
}
trap on_interrupt INT TERM

run_method() {
  local method="$1" limit="$2"
  shift 2
  printf '%s\n' "$*" > "evidence/${method}_command.txt"
  set +e
  setsid timeout --signal=TERM --kill-after=30s "${limit}s" "$@" > "evidence/${method}.log" 2>&1 &
  active_pid=$!
  active_starttick=$(awk '{print $22}' "/proc/$active_pid/stat")
  active_pgid=$(ps -o pgid= -p "$active_pid" | tr -d ' ')
  printf '%s %s %s\n' "$active_pid" "$active_starttick" "$active_pgid" > "evidence/${method}_identity.txt"
  wait "$active_pid"
  local status=$?
  set -e
  printf '%s\n' "$status" > "evidence/${method}_exit_code.txt"
  if ! verify_cleanup "$method"; then
    echo "${method}: cleanup unknown or failed" > evidence/failure.txt
    exit 90
  fi
  active_pid=''
  if (( status != 0 )); then
    echo "${method}: exited ${status}; bundle stopped" > evidence/failure.txt
    exit "$status"
  fi
  case "$method" in
    cellvit) test -s cellvit/cellvit_cells.json && test -s cellvit/cellvit_metadata.json ;;
    hovernet) test -s hovernet/hovernet_cells.json && test -s hovernet/hovernet_metadata.json ;;
  esac || { echo "${method}: output incomplete" > evidence/failure.txt; exit 92; }
  printf '{"method":"%s","status":"complete"}\n' "$method" > "evidence/${method}_complete.json.tmp"
  mv "evidence/${method}_complete.json.tmp" "evidence/${method}_complete.json"
}

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CELLVIT_CACHE=/opt/cellphenotyper/models/cellvit
run_method cellvit 900 singularity exec --nv --bind /mnt/sata_ssd:/mnt/sata_ssd "$container" \
  /opt/micromamba/envs/cellvit/bin/python /opt/cellphenotyper/bin/run_cellvitpp.py \
  --image "$source_case/input/he.ome.tif" --shift "$case_dir/code/shift.json" --outdir "$case_dir/cellvit" \
  --executable /opt/micromamba/envs/cellvit/bin/cellvit-inference --model HIPT --taxonomy pannuke \
  --gpu 0 --batch-size 2 --cpus 2 --memory-mb 8192 --ray-workers 1 --ray-worker-cpus 1

run_method hovernet 1200 singularity exec --nv --bind /mnt/sata_ssd:/mnt/sata_ssd "$container" \
  python /opt/cellphenotyper/bin/run_hovernet_monusac.py \
  --image "$source_case/input/he.ome.tif" --shift "$case_dir/code/shift.json" --outdir "$case_dir/hovernet" \
  --gpu 0 --target-mpp 0.25 --batch-size 8 --inference-workers 2 --postproc-workers 2 \
  --chunk-shape 4096 --tile-shape 2048

date -u +%Y-%m-%dT%H:%M:%SZ > evidence/stopped_utc.txt
printf '{"status":"complete","methods":["cellvit","hovernet"]}\n' > evidence/BUNDLE_COMPLETE.json.tmp
mv evidence/BUNDLE_COMPLETE.json.tmp evidence/BUNDLE_COMPLETE.json
