import re
import copy
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from PIL import Image as PILImage
import torch
import os
from transformers import AutoProcessor

# def build_llava_dataset(json_path, image_dir):
#     dataset = load_dataset("json", data_files=json_path, split="train")
#     def preprocess(example):
#         image_path = os.path.join(image_dir, example["image"])
#         example["images"] = [PILImage.open(image_path)]

#         conversations = example["conversations"]
#         result = []
#         first_question = conversations[0]["value"].replace("<image>\n", "")
#         result.append({
#             "content": [
#                 {"index": None, "text": first_question, "type": "text"},
#                 {"index": 0, "text": None, "type": "image"}
#             ],
#             "role": "user"
#         })
#         result.append({
#             "content": [
#                 {"index": None, "text": conversations[1]["value"], "type": "text"}
#             ],
#             "role": "assistant"
#         })
#         for i in range(2, len(conversations), 2):
#             result.append({
#                 "content": [
#                     {"index": None, "text": conversations[i]["value"], "type": "text"}
#                 ],
#                 "role": "user"
#             })
#             if i + 1 < len(conversations):
#                 result.append({
#                     "content": [
#                         {"index": None, "text": conversations[i + 1]["value"], "type": "text"}
#                     ],
#                     "role": "assistant"
#                 })
#         example["messages"] = result
#         del example["conversations"]
#         return example
#     dataset = dataset.map(preprocess)
#     return dataset

# def build_loader(
#     dataset,
#     batch_size,
#     num_workers,
#     processor,
#     ):

#     def collate_fn(examples):
#         # Get the texts and images, and apply the chat template
#         texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
#         images = [example["images"] for example in examples]
#         images = [image[0] for image in images]

#         # Tokenize the texts and process the images
#         batch = processor(images=images, text=texts, return_tensors="pt", padding=True)

#         # The labels are the input_ids, and we mask the padding tokens in the loss computation
#         labels = batch["input_ids"].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100  #
#         # Ignore the image token index in the loss computation (model specific)
#         image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
#         labels[labels == image_token_id] = -100
#         loss_mask = (labels != -100).long()
#         batch["labels"] = labels
#         batch["loss_mask"]=loss_mask
#         return batch
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size, 
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=num_workers
#     )

#     return dataloader

def build_llava_dataset(json_path, image_dir, processor, max_length):
    dataset = load_dataset("json", data_files=json_path, split="train")
    def preprocess(example):
        # Load image
        image_path = os.path.join(image_dir, example["image"])
        image = PILImage.open(image_path).convert("RGB")

        # Process messages into chat text
        conversations = example["conversations"]
        first_question = conversations[0]["value"].replace("<image>\n", "").replace("\n<image>", "")
        messages = [
            {
                "content": [
                    {"index": None, "text": first_question, "type": "text"},
                    {"index": 0, "text": None, "type": "image"},
                ],
                "role": "user"
            },
            {
                "content": [
                    {"index": None, "text": conversations[1]["value"], "type": "text"}
                ],
                "role": "assistant"
            }
        ]
        for i in range(2, len(conversations), 2):
            messages.append({
                "content": [
                    {"index": None, "text": conversations[i]["value"], "type": "text"}
                ],
                "role": "user"
            })
            if i + 1 < len(conversations):
                messages.append({
                    "content": [
                        {"index": None, "text": conversations[i + 1]["value"], "type": "text"}
                    ],
                    "role": "assistant"
                })
        # Convert message list to chat text
        chat_text = processor.apply_chat_template(messages, tokenize=False)
        text = copy.deepcopy(chat_text)
        # Preprocess image + tokenize
        processed = processor(
            images=image, 
            text=chat_text, 
            return_tensors="pt", 
            padding=True, 
            return_offsets_mapping=True,
            max_length=max_length
            )
        # Prepare labels
        input_ids = processed["input_ids"][0]
        labels = input_ids.clone()
        pad_token_id = processor.tokenizer.pad_token_id
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        num_image_token = (input_ids == image_token_id).sum().item()
        text = text.replace("<image>","<image>"*num_image_token)
    
        labels[labels == pad_token_id] = -100
        labels[labels == image_token_id] = -100

        # print(chat_text)
        offsets = processed["offset_mapping"][0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
        assistant_pattern = r'ASSISTANT:\s*(.*?)(?=USER:|$)'
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            start_char = match.start(1)
            end_char = match.end(1)
            for idx, (token_start, token_end) in enumerate(offsets.tolist()):
                
                if token_end <= start_char:
                    continue
                if token_start >= end_char:
                    continue
                loss_mask[idx] = 1

        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": processed["attention_mask"][0].unsqueeze(0),
            "pixel_values": processed["pixel_values"][0].unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "loss_mask": loss_mask.unsqueeze(0)
        }

    # dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        num_proc=32,
        desc="Preprocessing"
    )
    return dataset

def build_loader(dataset, batch_size, num_workers):
    def collate_fn(examples):
        return {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
            "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
            "labels": torch.stack([ex["labels"] for ex in examples]),
            "loss_mask": torch.stack([ex["loss_mask"] for ex in examples]),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return dataloader



if __name__ == "__main__":
    model_id = "/home/wulin/llm/SpecForge/cache/model/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)

    json_path = "/home/wulin/llm/SpecForge/cache/dataset/llava_sft/sampled2.json"
    image_dir = "/home/wulin/llm/SpecForge/cache/dataset/llava_sft/images"
    dataset = build_llava_dataset(json_path,image_dir, processor)
    print(dataset[0]["loss_mask"])

    # loader = build_loader(dataset,2,4)
    # batch = next(iter(loader))
    # print(batch)


