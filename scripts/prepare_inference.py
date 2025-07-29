import asyncio
import io
import os
import json
import argparse
import pandas as pd
import math

from PIL import Image
import requests
from tqdm import tqdm
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()

"""
This script will use llm to inference data for training:
{
    "id": str,
    "conversations": [
        {
            "role": str,
            "content": str
        }
    ],
}
"""

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

def main():
    args = parse_args()
    if args.dataset == "gsm8k":
        data = pd.read_parquet(args.data_path)
        data = data[args.start_id::args.split]
        llm = sgl.Engine(model_path=args.model_path)
        sampling_params = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 2048}
        batch_size = 32
        questions = [row["question"].strip() for _, row in data.iterrows()]
        with open(args.output_path, "w", encoding="utf-8") as fout:
            total_batches = math.ceil(len(questions) / batch_size)
            for i in tqdm(range(0, len(questions), batch_size), total=total_batches):
                batch = questions[i:i + batch_size]
                outputs = llm.generate(batch, sampling_params)

                for j, output in enumerate(outputs):
                    item = {
                        "id": i + j,
                        "conversations": [
                            {"role": "user", "content": batch[j]},
                            {"role": "assistant", "content": output["text"]},
                        ]
                    }
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    

if __name__ == "__main__":
    main()