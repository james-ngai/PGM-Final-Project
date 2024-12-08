
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 img_train.py \
    --config configs/img_eca-eval.yaml
