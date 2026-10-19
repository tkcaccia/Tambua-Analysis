#!/usr/bin/env bash
# One authorized Chiamaka GigaTIME bundle: 83 new MoNuSAC ROIs; reuse 37 verified outputs.
# Source ZIPs, prepared OME files and earlier failed comparison cases remain immutable.
set -Eeuo pipefail

case_dir=/mnt/sata_ssd/Tambua-monusac-matched-2026-10-19-v1
input_dir=/mnt/sata_ssd/Tambua-monusac-matched-2026-10-19-prep-v2
python_bin=/mnt/sata_ssd/Tambua-orion-2026-10-19-v1/.venv/bin/python
gpu_uuid=GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5
cd "$case_dir"
mkdir evidence/launch_guard
sha256sum -c evidence/staged_sha256.txt > evidence/hash_check.log
printf '%s\n' "$gpu_uuid" > evidence/assigned_gpu_uuid.txt
date -u +%Y-%m-%dT%H:%M:%SZ > evidence/started_utc.txt

timeout 10s nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > evidence/gpu_before.csv
if [[ -s evidence/gpu_before.csv ]]; then
  echo 'Assigned GPU busy before launch; bundle stopped' > evidence/failure.txt
  exit 70
fi

active_pid=''
active_pgid=''
active_starttick=''
active_sample=''
verify_cleanup() {
  local sample="$1" step
  for step in $(seq 1 15); do
    ps -eo pid=,pgid=,args= | awk -v pg="$active_pgid" '$2 == pg {print}' > "evidence/${sample}_process_after.txt"
    timeout 10s nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "evidence/${sample}_gpu_after.csv" || return 1
    if [[ ! -s "evidence/${sample}_process_after.txt" && ! -s "evidence/${sample}_gpu_after.csv" ]]; then
      echo 'Verified no owned process group member or assigned GPU compute context' > "evidence/${sample}_cleanup.txt"
      return 0
    fi
    sleep 2
  done
  echo 'Cleanup unknown or failed; bundle stopped' > "evidence/${sample}_cleanup.txt"
  return 1
}
stop_owned() {
  local current_tick
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
  if [[ -n "$active_sample" ]]; then verify_cleanup "$active_sample" || true; fi
}
on_interrupt() {
  trap - INT TERM
  echo 'Bundle interrupted; no later image may start' > evidence/failure.txt
  stop_owned
  exit 130
}
trap on_interrupt INT TERM

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
mapfile -t samples < input/NEW_SAMPLES.txt
if [[ "${#samples[@]}" != 83 ]]; then
  echo 'Frozen new-sample list is not 83 images; bundle stopped' > evidence/failure.txt
  exit 71
fi
deadline=$(( $(date +%s) + 2400 ))

for sample in "${samples[@]}"; do
  if (( $(date +%s) + 420 >= deadline )); then
    echo "Deadline margin reached before ${sample}; bundle stopped" > evidence/failure.txt
    exit 72
  fi
  active_sample="$sample"
  in="$input_dir/$sample"
  out="$case_dir/output/$sample"
  mkdir "$out"
  printf '%s\n' "$sample" > "evidence/${sample}_started.txt"
  set +e
  setsid timeout --signal=TERM --kill-after=30s 300s \
    "$python_bin" -u "$case_dir/code/run_gigatime_on_crop.py" \
    --image "$in/he.ome.tif" \
    --nuclei-mask "$in/expert_nuclei.ome.tif" \
    --cell-mask "$in/expert_anchored_cell_proxy.ome.tif" \
    --quant-dir "$out/quantification" --outdir "$out/gigatime_output" \
    --device cuda --repo-revision a6556030bdb8071857559e12b921620aa1bbac5b \
    --strict-target-mpp --target-mpp 0.25 --output-format none \
    --jpg-markers DAPI,CK,CD3,CD68,CD14,CD16 \
    --batch-size 2 --block-size 1024 > "evidence/${sample}.log" 2>&1 &
  active_pid=$!
  active_starttick=$(awk '{print $22}' "/proc/$active_pid/stat")
  active_pgid=$(ps -o pgid= -p "$active_pid" | tr -d ' ')
  printf '%s %s %s\n' "$active_pid" "$active_starttick" "$active_pgid" > "evidence/${sample}_identity.txt"
  wait "$active_pid"
  status=$?
  set -e
  printf '%s\n' "$status" > "evidence/${sample}_exit_code.txt"
  if ! verify_cleanup "$sample"; then
    echo "${sample}: cleanup unknown; bundle stopped" > evidence/failure.txt
    exit 90
  fi
  active_pid=''
  if (( status != 0 )); then
    echo "${sample}: exited ${status}; bundle stopped" > evidence/failure.txt
    exit "$status"
  fi
  "$python_bin" "$case_dir/code/validate_sample.py" "$in" "$out" > "evidence/${sample}_validation.log" 2>&1 || {
    echo "${sample}: validation failed; bundle stopped" > evidence/failure.txt
    exit 91
  }
  printf '{"sample_id":"%s","status":"complete"}\n' "$sample" > "$out/RUN_COMPLETE.json.tmp"
  mv "$out/RUN_COMPLETE.json.tmp" "$out/RUN_COMPLETE.json"
  active_sample=''
done

date -u +%Y-%m-%dT%H:%M:%SZ > evidence/stopped_utc.txt
printf '{"status":"complete","new_images":83,"reused_images":37}\n' > evidence/BUNDLE_COMPLETE.json.tmp
mv evidence/BUNDLE_COMPLETE.json.tmp evidence/BUNDLE_COMPLETE.json
