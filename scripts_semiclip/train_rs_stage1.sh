# RS-ALL L=U SemiCLIP
for seed in 0 1 2 ; do
    python -m torch.distributed.run --nproc_per_node 1 \
        --master_port 35471 \
        main.py -- \
        --model "ViT-B-16" \
        --pretrained openai \
        --train-data "RS-ALL" \
        --data-dir "./data/" \
        --label-ratio "0.1" \
        --val-data "RS-ALL" \
        --imagenet-val "RSICD-CLS" \
        --keyword-path "keywords/RS/class-name.txt" \
        --lr 5e-5 \
        --warmup 10 \
        --zeroshot-frequency 5 \
        --precision amp \
        --method "semiclip" \
        --seed "$seed" \
        --stage 1 \
        --epochs 25 \
        --save_ckpt \
        --pkname "stage1/rs_stage1_semiclip_seed${seed}" \
        --pratio 0.3 \
        --logs "./results/rs_stage1_semiclip_seed${seed}/"
done

# RS-ALL L/=U SemiCLIP
for seed in 0 1 2 ; do
    python -m torch.distributed.run --nproc_per_node 1 \
        --master_port 35472 \
        main.py -- \
        --model "ViT-B-16" \
        --pretrained openai \
        --train-data "RS-SHIFT" \
        --data-dir "./data/" \
        --label-ratio "0.1" \
        --val-data "RS-ALL" \
        --imagenet-val "RSICD-CLS" \
        --keyword-path "keywords/RS/class-name.txt" \
        --lr 5e-5 \
        --warmup 10 \
        --zeroshot-frequency 5 \
        --precision amp \
        --method "semiclip" \
        --seed "$seed" \
        --stage 1 \
        --epochs 25 \
        --save_ckpt \
        --pkname "stage1/rs_shift_stage1_semiclip_seed${seed}" \
        --pratio 0.3 \
        --logs "./results/rs_shift_stage1_semiclip_seed${seed}/"
done

