SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# train eagle3 for llava
# NUM_GPUS=${1:-8}
NUM_GPUS=6

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online_mllm.py \
    --target-model-path "$ROOT_DIR/cache/model/llava-1.5-7b-hf" \
    --draft-model-config "$ROOT_DIR/configs/llava_eagle3.json" \
    --train_json_path "$ROOT_DIR/cache/dataset/llava_sft/sampled30000.json" \
    --train_images_path "$ROOT_DIR/cache/dataset/llava_sft/images"\
    --output-dir "$ROOT_DIR/outputs/llava-eagle3" \
    --num-epochs 20 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "language_model.model.embed_tokens.weight"
