DATASET="gsm8k"
DATA_PATH="cache/dataset/gsm8k/main/train-00000-of-00001.parquet"
MODEL_PATH="cache/model/Qwen3-8B"
OUTPUT_DIR="cache/dataset/gsm8k/main"
SPLIT=4

for i in $(seq 0 $((SPLIT - 1))); do
  export CUDA_VISIBLE_DEVICES=$i
  echo "Launching split $i on GPU $i..."
  python scripts/prepare_inference.py \
    --dataset $DATASET \
    --output_path ${OUTPUT_DIR}/train_${i}.jsonl \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --start_id $i \
    --split $SPLIT &
done

wait
echo "All inference preparations completed."
