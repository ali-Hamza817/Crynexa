#!/bin/bash
# Remaining experiment matrix. --skip-done makes every stage resumable.
cd /home/administrator/Desktop/Crynexa
log(){ echo "[$(date +%H:%M:%S)] $*" >> runs/pipeline.log; }
GPUS="${GPUS:-0,2,1}"

for spec in "E2:cifar10:_cifar" "E2:shapes:_shapesB" "E1:shapes:_shapesB" "E1:cifar10:_cifar"; do
  IFS=: read -r suite ds tag <<< "$spec"
  log "$suite $ds starting (gpus $GPUS)"
  python3 sweep.py --suite "$suite" --gpus "$GPUS" --dataset "$ds" --tag "$tag" \
    --steps 5000 --skip-done >> "runs/sweep_${suite}${tag}.log" 2>&1
  log "$suite $ds done"
done

python3 build_web.py >/dev/null 2>&1
log "PIPELINE COMPLETE"
