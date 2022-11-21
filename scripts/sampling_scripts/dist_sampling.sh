export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NNODE=3
torchrun \
    --nnodes=$NNODE \
    --nproc_per_node 8 \
    --rdzv_id v9_dist_sample \
    --rdzv_backend c10d \
    --rdzv_endpoint $1:29500 \
    sampling.py -c $2 --nnode $NNODE -e $3 -n $4

# usage: bash scripts/sampling_scripts/dist_sampling.sh \
# 172.31.0.139 configs/laion_cocostuff_text_v9.json \
# 59 5 # this is machine 1 ip address
