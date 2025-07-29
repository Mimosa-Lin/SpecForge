# python3 -m sglang.launch_server \
#     --model cache/model/Qwen3-8B \
#     --mem-fraction-static 0.75 \
#     --cuda-graph-max-bs 2 \
#     --tp 1 \
#     --context-length 2048 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 30000 \
#     --dtype bfloat16


MODEL_PATH="cache/model/Qwen2.5-VL-7B-Instruct"
HOST="0.0.0.0"
BASE_PORT=30000
NUM_GPUS=6

for ((i=0; i<$NUM_GPUS; i++))
do
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --port $((BASE_PORT + i)) \
        --host $HOST \
        --context-length 2048 \
        --max-running-requests 1 \
        --mem-fraction-static 0.5 \
        --tensor-parallel-size 1 \
        --chat-template qwen2-vl \
        > logs/server_gpu$i.log 2>&1 &
    echo "Started server on GPU $i at port $((BASE_PORT + i))"
done

wait

