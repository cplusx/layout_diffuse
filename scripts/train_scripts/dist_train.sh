export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NNODE=6
torchrun \
    --nnodes=$NNODE \
    --nproc_per_node 4 \
    --rdzv_id v9_dist \
    --rdzv_backend c10d \
    --rdzv_endpoint $1:29500 \
    main.py -c $2 -n $NNODE -r

# usage: bash scripts/train_scripts/dist_train.sh 172.31.42.68