from transformers import pipeline

import csv
import time
from multiprocessing.pool import ThreadPool
import requests
import os
import PIL.Image as Image
from io import BytesIO
import numpy as np
from tqdm import tqdm
import shutil
from transformers import AutoTokenizer, GenerationConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
import ast
import re

device = "cuda:0"
base_file = "./data/eval_images_coarse-grained.csv"
model = "microsoft/Promptist"
tokenizer = AutoTokenizer.from_pretrained(model)
gpt2_pipe = pipeline("text-generation", model=model, device=device)


def pre_caption(caption, max_words=256):
    caption = (
        re.sub(
            # r"([,.'!?\"()*#:;~])",
            r"([.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if caption_words[0].lower() == "shein":
        caption_words = caption_words[1:]
    if max_words is not None and len(caption_words) > max_words:
        caption_words = caption_words[:max_words]
    caption = " ".join(caption_words)
    # caption不能为空
    if caption == "":
        caption = "a cloth"
    # if not caption.isascii():
    #     # caption = 'a product'
    # print(caption)
    return caption


output = []
with open(base_file, "r+") as f:
    data = csv.DictReader(f)
    data = [r for r in data]
    data = data[:1000]
    pbar = tqdm(total=len(data))
    for row in data:
        prompt_list = ast.literal_eval(row["coarse-grained prompt"])
        prompt = prompt_list[1]["prompt"]
        # for i, _ in enumerate(gpt2_ans):
        #     row[f""] = {"prompt": pre_caption(_["generated_text"].strip()).encode("utf-8")}
        row["prompt_easy"] = prompt
        output.append(row)
        pbar.update(1)
    pbar.close()

with open(f"./data_comparison/eval_chatgpt.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)
