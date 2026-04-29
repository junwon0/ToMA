# fashion SemiCLIP
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
        --stage 1 \
        --epochs 25 \
        --save_ckpt \
        --pkname "stage1_fashion/fashion_stage1_semiclip_seed${seed}" \
        --pratio 0.3 \
        --logs "./results/fashion_stage1_semiclip_seed${seed}/"
done
