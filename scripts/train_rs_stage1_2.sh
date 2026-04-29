
for topo_reg in "ToMA" ; do
    for bs in 128 ; do
        for scale in 0.5 ; do
            # RS-ALL L=U ToMA
            for seed in 0 1 2 ; do
                python -m torch.distributed.run --nproc_per_node 1 \
                    --master_port 35472 \
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
                    --batch-size "$bs" \
                    --stage 1 \
                    --epochs 25 \
                    --save_ckpt \
                    --pkname "stage1/rs_stage1_${topo_reg}_bs${bs}_${scale}_seed${seed}" \
                    --pratio 0.3 \
                    --logs "./results/rs_stage1_${topo_reg}_bs${bs}_${scale}_seed${seed}/" \
                    --topo-reg-type "$topo_reg" \
                    --topo-reg-stage1 "image-text" \
                    --topo-reg-scale "${scale}"
            done

            for seed in 0 1 2; do
                python -m torch.distributed.run --nproc_per_node 1 \
                    --master_port 35472 \
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
                    --topo-reg-type "$topo_reg" \
                    --topo-reg-stage2 "image-text-mix" \
                    --seed "$seed" \
                    --batch-size "$bs" \
                    --stage 2 \
                    --epochs 15 \
                    --pkname "stage1/rs_stage1_${topo_reg}_bs${bs}_${scale}_seed${seed}" \
                    --pratio 0.3 \
                    --save_ckpt \
                    --logs "./results/rs_stage1_${topo_reg}_bs${bs}_${scale}_seed${seed}_stage2/stage2_${topo_reg}_bs${bs}_${scale}_seed${seed}/" \
                    --resume-path "./results/rs_stage1_${topo_reg}_bs${bs}_${scale}_seed${seed}/checkpoints" \
                    --topo-reg-scale "${scale}"
            done

            ##########################################################################################
            # Use domain information #################################################################
            ##########################################################################################

            # RS-ALL L=U ToMA
            for seed in 0 1 2 ; do
                python -m torch.distributed.run --nproc_per_node 1 \
                    --master_port 35472 \
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
                    --batch-size "$bs" \
                    --stage 1 \
                    --epochs 25 \
                    --save_ckpt \
                    --pkname "stage1/rs_stage1_${topo_reg}_bs${bs}_domain_${scale}_seed${seed}" \
                    --pratio 0.3 \
                    --logs "./results/rs_stage1_${topo_reg}_bs${bs}_domain_${scale}_seed${seed}/" \
                    --topo-reg-type "$topo_reg" \
                    --topo-reg-stage1 "image-text" \
                    --sampling-data-domain True \
                    --topo-reg-scale "${scale}"
            done

            for seed in 0 1 2; do
                python -m torch.distributed.run --nproc_per_node 1 \
                    --master_port 35472 \
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
                    --topo-reg-type "$topo_reg" \
                    --topo-reg-stage2 "image-text-mix" \
                    --seed "$seed" \
                    --batch-size "$bs" \
                    --stage 2 \
                    --epochs 15 \
                    --pkname "stage1/rs_stage1_${topo_reg}_bs${bs}_domain_${scale}_seed${seed}" \
                    --pratio 0.3 \
                    --save_ckpt \
                    --logs "./results/rs_stage1_${topo_reg}_bs${bs}_domain_${scale}_seed${seed}_stage2/stage2_${topo_reg}_bs${bs}_domain_${scale}_seed${seed}/" \
                    --resume-path "./results/rs_stage1_${topo_reg}_bs${bs}_domain_${scale}_seed${seed}/checkpoints" \
                    --sampling-data-domain True \
                    --topo-reg-scale "${scale}"
            done
            ###################################################################################################
        done
    done
done