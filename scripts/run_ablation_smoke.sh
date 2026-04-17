#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/smoke.yaml"

run_and_capture() {
  local name="$1"
  shift
  local output
  if [[ -x ".venv/bin/python" ]]; then
    output=$(.venv/bin/python -m fractal_swin_unet.train --config "$CONFIG" --use_synth_data "$@")
  else
    output=$(python3 -m fractal_swin_unet.train --config "$CONFIG" --use_synth_data "$@")
  fi
  echo "$output" | grep "FINAL_METRICS" | tail -n 1 | awk -v name="$name" '{print name, $2, $3}'
}

printf "Model\tTrainDice\tValDice\n"
run_and_capture "baseline" --set model.name=baseline --set train.epochs=1 --set train.steps_per_epoch=1 --set train.batch_size=2
run_and_capture "fractal_prior" --set model.name=fractal_prior --set model.enable_fractal_gate=true --set train.epochs=1 --set train.steps_per_epoch=1 --set train.batch_size=2
