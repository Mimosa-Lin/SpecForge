DATASET="llava_sft"
DATA_PATH="cache/dataset/llava_sft"
MODEL_PATH="cache/model/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="cache/dataset/llava_sft"
SPLIT=6

for i in $(seq 0 $((SPLIT - 1))); do
  export CUDA_VISIBLE_DEVICES=$i
  python scripts/prepare_mllm_data.py \
    --dataset $DATASET \
    --output_path ${OUTPUT_DIR}/train_${i}.jsonl \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --start_id $i \
    --split $SPLIT &
done

wait
echo "All inference preparations completed."
