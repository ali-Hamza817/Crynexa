#!/bin/bash
cd /home/administrator/Desktop/Crynexa
r(){ python3 run_experiment.py --name "$1" --device "$2" --json "$3" > runs/$1.log 2>&1; echo "$1 rc=$?" >> runs/val2_status.txt; }
: > runs/val2_status.txt
B='"dataset":"noise","n_keys":1,"steps":8000,"eval_every":1000,"eval_n":1000,"workers":6,"input_repr":"both"'
r v2_cbc_diff  cuda:0 "{$B,\"cipher\":{\"mode\":\"diffuse_only\",\"rounds\":1,\"feedback\":\"cbc\"}}" &
r v2_full_r2   cuda:2 "{$B,\"cipher\":{\"mode\":\"permute_diffuse\",\"rounds\":2,\"feedback\":\"cbc\"}}" &
wait
r v2_permonly  cuda:0 "{$B,\"cipher\":{\"mode\":\"permute_only\",\"rounds\":1}}" &
r v2_full_r1   cuda:2 "{$B,\"cipher\":{\"mode\":\"permute_diffuse\",\"rounds\":1,\"feedback\":\"cbc\"}}" &
wait
echo ALLDONE >> runs/val2_status.txt
