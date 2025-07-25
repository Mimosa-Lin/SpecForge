export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)


python \
    $ROOT_DIR/scripts/train_eagle3_online_single.py \
    --target-model-path "$ROOT_DIR/cache/model/Qwen3-8B" \
    --draft-model-config "$ROOT_DIR/configs/qwen3-8B-eagle3.json" \
    --train-data-path "$ROOT_DIR/cache/dataset/gsm8k/main/train.jsonl" \
    --output-dir "$ROOT_DIR/outputs/Qwen3-8B-eagle3" \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template Qwen3 \
    --cache-dir $ROOT_DIR/cache
