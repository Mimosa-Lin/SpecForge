export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --resume-download bpietroiu/qwen2.5-0.5B-speculative-padded --local-dir model_checkpoints/qwen2.5-0.5B-speculative-padded
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir cache/model/Qwen2.5-VL-7B-Instruct
# huggingface-cli download --repo-type dataset --resume-download openai/gsm8k --local-dir cache/dataset/gsm8k

