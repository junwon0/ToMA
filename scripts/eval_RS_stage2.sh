# ckpt=$1

ckpts=(

    "rs_stage1_semiclip_seed0_stage2/stage2_semiclip_seed0"
    "rs_stage1_semiclip_seed1_stage2/stage2_semiclip_seed1"
    "rs_stage1_semiclip_seed2_stage2/stage2_semiclip_seed2"
    
    "rs_shift_stage1_semiclip_seed0_stage2/stage2_semiclip_seed0"
    "rs_shift_stage1_semiclip_seed1_stage2/stage2_semiclip_seed1"
    "rs_shift_stage1_semiclip_seed2_stage2/stage2_semiclip_seed2"

    
    "rs_stage1_ToMA_bs128_0.5_seed0_stage2/stage2_ToMA_bs128_0.5_seed0"
    "rs_stage1_ToMA_bs128_0.5_seed1_stage2/stage2_ToMA_bs128_0.5_seed1"
    "rs_stage1_ToMA_bs128_0.5_seed2_stage2/stage2_ToMA_bs128_0.5_seed2"
    "rs_stage1_ToMA_bs128_domain_0.5_seed0_stage2/stage2_ToMA_bs128_domain_0.5_seed0"
    "rs_stage1_ToMA_bs128_domain_0.5_seed1_stage2/stage2_ToMA_bs128_domain_0.5_seed1"
    "rs_stage1_ToMA_bs128_domain_0.5_seed2_stage2/stage2_ToMA_bs128_domain_0.5_seed2"

    "rs_shift_stage1_ToMA_bs128_0.5_seed0_stage2/stage2_ToMA_bs128_0.5_seed0"
    "rs_shift_stage1_ToMA_bs128_0.5_seed1_stage2/stage2_ToMA_bs128_0.5_seed1"
    "rs_shift_stage1_ToMA_bs128_0.5_seed2_stage2/stage2_ToMA_bs128_0.5_seed2"
    "rs_shift_stage1_ToMA_bs128_domain_0.5_seed0_stage2/stage2_ToMA_bs128_domain_0.5_seed0"
    "rs_shift_stage1_ToMA_bs128_domain_0.5_seed1_stage2/stage2_ToMA_bs128_domain_0.5_seed1"
    "rs_shift_stage1_ToMA_bs128_domain_0.5_seed2_stage2/stage2_ToMA_bs128_domain_0.5_seed2"

)

ZEROSHOT_DATASETS=(
"RSICD-CLS"
"UCM-CLS"
"WHU-RS19"
"RSSCN7"
"AID"
)
RETRIEVAL_DATASETS=(
"RSICD"
"UCM"
"Sydney"
)

# zero-shot classification
for ckpt in "${ckpts[@]}"; do
    for dataset in "${ZEROSHOT_DATASETS[@]}"; do
        python main.py \
            --model "ViT-B-16" \
            --name ${ckpt} \
            --data-dir "./data/" \
            --imagenet-val ${dataset} \
            --stage 2 \
    #
    done
done
# image-text retrieval
for ckpt in "${ckpts[@]}"; do
    for dataset in "${RETRIEVAL_DATASETS[@]}"; do
        python main.py \
            --model "ViT-B-16" \
            --name ${ckpt} \
            --data-dir "./data/" \
            --val-data ${dataset} \
            --stage 2 \
    #
    done
done
