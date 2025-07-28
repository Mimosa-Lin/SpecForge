export CUDA_VISIBLE_DEVICES=0

python3 -m sglang.launch_server \
    --model cache/model/Qwen3-8B  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path outputs/Qwen3-8B-eagle3/epoch_19 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 1 \
    --context-length 2048 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16