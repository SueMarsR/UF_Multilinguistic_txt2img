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
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
import ast
import re

device = "cuda:0"
base_file = "./data/eval_images_coarse-grained.csv"
model = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model)
# pip_model = T5ForConditionalGeneration.from_pretrained(model)
t5 = T5ForConditionalGeneration.from_pretrained(model)
t5.to(device)
tokenizer = T5Tokenizer.from_pretrained(model)
# gpt2_pipe = pipeline("text-generation", model=pip_model, tokenizer=tokenizer, device=device)


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
        # gpt2_ans = gpt2_pipe(
        #     prompt,
        #     # do_sample=True,
        #     # num_return_sequences=5,num_beams=10,
        #     do_sample=True,
        #     top_k=50,
        #     top_p=0.95,
        #     num_return_sequences=1,
        #     # skip_special_tokens=True,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        prefix = "continue this sentence: "
        context = prompt
        input_seq = prefix + '"' + context + "..." + '"'
        # 使用分词器进行编码
        input_ids = tokenizer.encode(input_seq, return_tensors="pt").to(device)
        # input_ids.to(device)
        outputs = t5.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        ).cpu()
        output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        _ = context + output_str
        # print(_)
        row[f"ans_0"] = {"prompt": pre_caption(_)}
        output.append(row)
        pbar.update(1)
    pbar.close()

with open(f"./data_comparison/eval_images_coarse-graied_flan-t5-base.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)
