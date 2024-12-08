
# export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 train.py \
    --config configs/eca-eval.yaml
