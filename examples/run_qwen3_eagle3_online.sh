SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export CUDA_VISIBLE_DEVICES=0
# train eagle3 for qwen3-8b
# NUM_GPUS=${1:-8}
NUM_GPUS=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path "$ROOT_DIR/cache/model/Qwen3-8B" \
    --draft-model-config "$ROOT_DIR/configs/qwen3-8B-eagle3.json" \
    --train-data-path "$ROOT_DIR/cache/dataset/gsm8k/main/train.jsonl" \
    --output-dir "$ROOT_DIR/outputs/Qwen3-8B-eagle3" \
    --num-epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache
