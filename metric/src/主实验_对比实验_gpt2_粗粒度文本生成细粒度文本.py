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

device = "cuda:1"
base_file = "./data/eval_images_coarse-grained.csv"
model = "gpt2"
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
    pbar = tqdm(total=len(data))
    for row in data:
        prompt_list = ast.literal_eval(row["coarse-grained prompt"])
        prompt = prompt_list[1]["prompt"]
        gpt2_ans = gpt2_pipe(
            prompt,
            # do_sample=True,
            # num_return_sequences=5,num_beams=10,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            # skip_special_tokens=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        for i, _ in enumerate(gpt2_ans):
            row[f"ans_{i}"] = {"prompt": pre_caption(_["generated_text"].strip())}
        output.append(row)
        pbar.update(1)
    pbar.close()

with open(f"./data_comparison/eval_images_coarse-graied_gpt2.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)
