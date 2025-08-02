import os
import json
from PIL import Image
from tqdm import tqdm
import time
import torch

from transformers import AutoProcessor
from specforge.hf_model import LlavaForConditionalGeneration
from specforge.eagle3 import LlamaForCausalLMEagle3, EaModel

def naivegenerate(model, ids, problems):
    times = []
    ids = [id for id in ids if id in os.listdir(image_folder)]
    for id in tqdm(ids,total=len(ids)):
        problem = problems[id]
        image_path = os.path.join(image_folder, id, "image.png")
        if not os.path.exists(image_path):
            continue 
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": problem["question"]},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(image_path)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)
        start_time = time.time()
        out = model.naivegenerate(
            input_ids=inputs["input_ids"],
            pixel_values = inputs["pixel_values"],
        )
        end_time = time.time()
        times.append(end_time - start_time)
        del out, inputs
    avg_time = sum(times) / len(times)
    print(f"Average generation time: {avg_time:.4f} seconds")
    return avg_time

def eagenerate(model, ids, problems):
    times = []
    average_accept_lengths = []
    ids = [id for id in ids if id in os.listdir(image_folder)]
    for id in tqdm(ids,total=len(ids)):
        image_path = os.path.join(image_folder, id, "image.png")

        if not os.path.exists(image_path):
            continue 
        problem = problems[id]
        
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": problem["question"]},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(image_path)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)
        start_time = time.time()
        out = model.eagenerate(
            input_ids=inputs["input_ids"],
            pixel_values = inputs["pixel_values"],
        )
        end_time = time.time()
        average_accept_lengths.append(out[-1])
        times.append(end_time - start_time)
        del out, inputs
    avg_time = sum(times) / len(times)
    avg_accept = sum(average_accept_lengths) / len(average_accept_lengths)
    print(f"Average generation time: {avg_time:.4f} seconds")
    print(f"Accepted tokens: {avg_accept+1:.4f}")
    return avg_time


if __name__ == "__main__":
    image_folder = "cache/dataset/ScienceQA/images/test"
    split_path = "cache/dataset/ScienceQA/split.json"
    problem_path = "cache/dataset/ScienceQA/problems.json"
    ids = json.load(open(split_path,"r"))
    problems = json.load(open(problem_path,"r"))
    
    model_id = "cache/model/llava-1.5-7b-hf"
    target_model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to("cuda:0", dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)
    draft_model = LlamaForCausalLMEagle3.from_pretrained("outputs/llava-eagle3_sampled30000/epoch_7").to("cuda:0", dtype=torch.float16)
    model = EaModel(target_model=target_model, draft_model=draft_model, tokenizer=processor.tokenizer)

    conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What could be a mistake to avoid while preparing and baking this pizza?"},
          {"type": "image"},
        ],
    },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image_file = "/home/wulin/llm/SpecForge/cache/dataset/llava_sft/images/000000223844.jpg"
    raw_image = Image.open(image_file)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)
    out = model(
        input_ids=inputs["input_ids"],
        pixel_values = inputs["pixel_values"],
        attention_mask = inputs["attention_mask"],
    )
    # naivegenerate(model, ids, problems)
    eagenerate(model, ids, problems)

    # out = model.naivegenerate(
    #     input_ids=inputs["input_ids"],
    #     pixel_values = inputs["pixel_values"],
    # )
    # print("naivegenerate output:",processor.decode(out, skip_special_tokens=True))

    # out = model.eagenerate(
    #     input_ids=inputs["input_ids"],
    #     pixel_values = inputs["pixel_values"],
    # )
    # print("eagenerate output:",processor.decode(out, skip_special_tokens=True))

