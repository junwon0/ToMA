# ckpt=$1
# fill this list before evaluation
ckpts=(

)

ZEROSHOT_DATASETS=(
"Fashion200k-CLS"
"Fashion200k-SUBCLS"
"FashionGen-CLS"
"FashionGen-SUBCLS"
"Polyvore-CLS"
)
RETRIEVAL_DATASETS=(
"Fashion200k"
"FashionGen"
"Polyvore"
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
