#!/bin/bash
cd /home/administrator/Desktop/Crynexa
r(){ python3 run_experiment.py --name "$1" --device "$2" --json "$3" > runs/$1.log 2>&1; echo "$1 rc=$?" >> runs/cal_status.txt; }
: > runs/cal_status.txt
B='"n_keys":1,"steps":6000,"eval_every":1000,"eval_n":1000,"workers":6,"input_repr":"both"'
r cal_shapes_full_r2 cuda:0 "{$B,\"dataset\":\"shapes\",\"cipher\":{\"mode\":\"permute_diffuse\",\"rounds\":2,\"feedback\":\"cbc\"}}" &
r cal_shapes_cbcdiff cuda:2 "{$B,\"dataset\":\"shapes\",\"cipher\":{\"mode\":\"diffuse_only\",\"rounds\":1,\"feedback\":\"cbc\"}}" &
wait
r cal_shapes_permonly cuda:0 "{$B,\"dataset\":\"shapes\",\"cipher\":{\"mode\":\"permute_only\",\"rounds\":1}}" &
r cal_noise_cbcdiff_pe cuda:2 "{$B,\"dataset\":\"noise\",\"cipher\":{\"mode\":\"diffuse_only\",\"rounds\":1,\"feedback\":\"cbc\"}}" &
wait
echo ALLDONE >> runs/cal_status.txt
