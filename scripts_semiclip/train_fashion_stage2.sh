
for seed in 0 1 2 ; do
    python -m torch.distributed.run --nproc_per_node 1 \
        --master_port 35470 \
        main.py -- \
        --model "ViT-B-16" \
        --pretrained openai \
        --train-data "Fashion-ALL" \
        --data-dir "./data/" \
        --label-ratio "0.1" \
        --val-data "Polyvore" \
        --imagenet-val "Polyvore-CLS" \
        --keyword-path "keywords/fashion/class-name.txt" \
        --lr 5e-5 \
        --warmup 10 \
        --zeroshot-frequency 5 \
        --precision amp \
        --method "semiclip" \
        --seed "$seed" \
        --stage 2 \
        --epochs 15 \
        --pkname "stage1_fashion/fashion_stage1_semiclip_seed${seed}" \
        --pratio 0.3 \
        --save_ckpt \
        --logs "./results/fashion_stage1_semiclip_seed${seed}_stage2/stage2_semiclip_seed${seed}/" \
        --resume-path "./results/fashion_stage1_semiclip_seed${seed}/checkpoints"
done