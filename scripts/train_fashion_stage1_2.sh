# fashion ToMA
for topo_reg in "ToMA" ; do
    for scale in 0.1 ; do
        for seed in 0 1 2 ; do
            python -m torch.distributed.run --nproc_per_node 1 \
                --master_port 35471 \
                main.py -- \
                --model "ViT-B-16" \
                --pretrained openai \
                --train-data "Fashion-ALL" \
                --data-dir "./data/" \
                --label-ratio "0.1" \
                --val-data "Fashion200k" \
                --imagenet-val "Fashion200k-CLS" \
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
                --pkname "stage1_fashion/fashion_stage1_${topo_reg}_${scale}_seed${seed}" \
                --pratio 0.3 \
                --logs "./results/fashion_stage1_${topo_reg}_${scale}_seed${seed}/" \
                --topo-reg-type "$topo_reg" \
                --topo-reg-stage1 "image-text" \
                --topo-reg-scale "${scale}"
        done

        for seed in 0 1 2 ; do
            python -m torch.distributed.run --nproc_per_node 1 \
                --master_port 35471 \
                main.py -- \
                --model "ViT-B-16" \
                --pretrained openai \
                --train-data "Fashion-ALL" \
                --data-dir "./data/" \
                --label-ratio "0.1" \
                --val-data "Fashion200k" \
                --imagenet-val "Fashion200k-CLS" \
                --keyword-path "keywords/fashion/class-name.txt" \
                --lr 5e-5 \
                --warmup 10 \
                --zeroshot-frequency 5 \
                --precision amp \
                --method "semiclip" \
                --seed "$seed" \
                --stage 2 \
                --epochs 15 \
                --save_ckpt \
                --pkname "stage1_fashion/fashion_stage1_${topo_reg}_${scale}_seed${seed}" \
                --pratio 0.3 \
                --logs "./results/fashion_stage1_${topo_reg}_${scale}_seed${seed}_stage2/stage2_${topo_reg}_${scale}_seed${seed}" \
                --resume-path "./results/fashion_stage1_${topo_reg}_${scale}_seed${seed}/checkpoints" \
                --topo-reg-type "$topo_reg" \
                --topo-reg-stage2 "image-text-mix" \
                --topo-reg-scale "${scale}"
        done

        ##########################################################################################
        # Use domain information #################################################################
        ##########################################################################################

        for seed in 0 1 2 ; do
            python -m torch.distributed.run --nproc_per_node 1 \
                --master_port 35471 \
                main.py -- \
                --model "ViT-B-16" \
                --pretrained openai \
                --train-data "Fashion-ALL" \
                --data-dir "./data/" \
                --label-ratio "0.1" \
                --val-data "Fashion200k" \
                --imagenet-val "Fashion200k-CLS" \
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
                --pkname "stage1_fashion/fashion_stage1_${topo_reg}_domain_${scale}_seed${seed}" \
                --pratio 0.3 \
                --logs "./results/fashion_stage1_${topo_reg}_domain_${scale}_seed${seed}/" \
                --topo-reg-type "$topo_reg" \
                --topo-reg-stage1 "image-text" \
                --topo-reg-scale "${scale}" \
                --sampling-data-domain True
        done

        for seed in 0 1 2 ; do
            python -m torch.distributed.run --nproc_per_node 1 \
                --master_port 35471 \
                main.py -- \
                --model "ViT-B-16" \
                --pretrained openai \
                --train-data "Fashion-ALL" \
                --data-dir "./data/" \
                --label-ratio "0.1" \
                --val-data "Fashion200k" \
                --imagenet-val "Fashion200k-CLS" \
                --keyword-path "keywords/fashion/class-name.txt" \
                --lr 5e-5 \
                --warmup 10 \
                --zeroshot-frequency 5 \
                --precision amp \
                --method "semiclip" \
                --seed "$seed" \
                --stage 2 \
                --epochs 15 \
                --save_ckpt \
                --pkname "stage1_fashion/fashion_stage1_${topo_reg}_domain_${scale}_seed${seed}" \
                --pratio 0.3 \
                --logs "./results/fashion_stage1_${topo_reg}_domain_${scale}_seed${seed}_stage2/stage2_${topo_reg}_domain_${scale}_seed${seed}" \
                --resume-path "./results/fashion_stage1_${topo_reg}_domain_${scale}_seed${seed}/checkpoints" \
                --topo-reg-stage2 "image-text-mix" \
                --topo-reg-type "$topo_reg" \
                --topo-reg-scale "${scale}" \
                --sampling-data-domain True
        done
    done
done
