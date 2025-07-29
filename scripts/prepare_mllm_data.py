import requests
import os
import json
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "llava_sft"],
        help="The demo dataset to quickly run the training for speculative decoding",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to save the processed dataset, if not specified, the dataset will be saved in the cache/dataset/dataset_name directory of the root path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the custom dataset, if not specified, the default dataset will be loaded",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--split",
        type=int,
        required=True,
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    port = 30000+args.start_id
    url = f"http://localhost:{port}/v1/chat/completions"
    json_path = os.path.join(args.data_path, "llava_instruct_150k.json")
    image_path = os.path.join(args.data_path, "images")
    data = json.load(open(json_path,"r"))[args.start_id::args.split]
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(data))):
            inputs = {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": data[i]["conversations"][0]["value"].replace("<image>\n","").replace("\n<image>","")},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": os.path.join(image_path,data[i]["image"])
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0,
            }

            response = requests.post(url, json=inputs)
            item = {
                "id": data[i]["id"],
                "image": data[i]["image"],
                "conversations": [
                    {
                        "from": "human",
                        "value": data[i]["conversations"][0]["value"]
                    },
                    {
                        "from": "gpt",
                        "value": json.loads(response.text)["choices"][0]["message"]["content"]
                    },
                ]
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

