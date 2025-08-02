from PIL import Image

import torch

from transformers import AutoProcessor
from specforge.hf_model import LlavaForConditionalGeneration
from specforge.eagle3 import LlamaForCausalLMEagle3, EaModel



if __name__ == "__main__":

    model_id = "cache/model/llava-1.5-7b-hf"
    model_id = "cache/model/llava-1.5-7b-hf"
    target_model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to("cuda:0", dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)
    draft_model = LlamaForCausalLMEagle3.from_pretrained("outputs/llava-eagle3_3000/epoch_10").to("cuda:0", dtype=torch.float16)
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
    # out = model(
    #     input_ids=inputs["input_ids"],
    #     pixel_values = inputs["pixel_values"],
    #     attention_mask = inputs["attention_mask"],
    # )
    out = model.naivegenerate(
        input_ids=inputs["input_ids"],
        pixel_values = inputs["pixel_values"],
    )
    print("naivegenerate output:",processor.decode(out, skip_special_tokens=True))

    image_file = "/home/wulin/llm/SpecForge/cache/dataset/llava_sft/images/000000223844.jpg"
    raw_image = Image.open(image_file)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)

    out = model.eagenerate(
        input_ids=inputs["input_ids"],
        pixel_values = inputs["pixel_values"],
    )
    print("eagenerate output:",processor.decode(out, skip_special_tokens=True))

