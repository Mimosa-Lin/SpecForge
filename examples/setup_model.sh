python3 -m sglang.launch_server \
    --model cache/model/llava-v1.5-7b \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 1 \
    --context-length 2048 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16


# MODEL_PATH="cache/model/llava-1.5-7b-hf"
# HOST="0.0.0.0"
# BASE_PORT=30000
# NUM_GPUS=1

# CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
#     --model-path $MODEL_PATH \
#     --port 30000 \
#     --host $HOST \
#     --context-length 2048 \
#     --max-running-requests 1 \
#     --mem-fraction-static 0.5 \
#     --tensor-parallel-size 1 \


