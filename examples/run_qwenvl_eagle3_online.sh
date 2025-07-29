SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# train eagle3 for qwen3-8b
# NUM_GPUS=${1:-8}
NUM_GPUS=6

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online_mllm.py \
    --target-model-path "$ROOT_DIR/cache/model/Qwen2.5-VL-7B-Instruct" \
    --draft-model-config "$ROOT_DIR/configs/qwenvl7b-eagle3.json" \
    --train_json_path "$ROOT_DIR/cache/dataset/llava_sft/merged.json"\
    --train_images_path "$ROOT_DIR/cache/dataset/llava_sft/images"\
    --output-dir "$ROOT_DIR/outputs/Qwen2.5-VL-7B-Instruct" \
    --num-epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --max-length 1024 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache
