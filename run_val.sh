#!/bin/bash
cd /home/administrator/Desktop/Crynexa
r(){ python3 run_experiment.py --name "$1" --device "$2" --json "$3" > runs/$1.log 2>&1; echo "$1 rc=$?" >> runs/val_status.txt; }
: > runs/val_status.txt
r val_xor_bits    cuda:0 '{"dataset":"noise","n_keys":1,"steps":4000,"eval_every":500,"eval_n":1000,"workers":6,"input_repr":"bits","cipher":{"mode":"diffuse_only","rounds":1,"feedback":"none"}}' &
r val_xor_pixel   cuda:2 '{"dataset":"noise","n_keys":1,"steps":4000,"eval_every":500,"eval_n":1000,"workers":6,"input_repr":"pixel","cipher":{"mode":"diffuse_only","rounds":1,"feedback":"none"}}' &
wait
r val_cbc_bits    cuda:0 '{"dataset":"noise","n_keys":1,"steps":4000,"eval_every":500,"eval_n":1000,"workers":6,"input_repr":"bits","cipher":{"mode":"diffuse_only","rounds":1,"feedback":"cbc"}}' &
r val_perm_mixer  cuda:2 '{"dataset":"noise","n_keys":1,"steps":4000,"eval_every":500,"eval_n":1000,"workers":6,"input_repr":"bits","model":"pixelmixer","batch_size":128,"cipher":{"mode":"permute_only","rounds":1}}' &
wait
echo ALLDONE >> runs/val_status.txt
