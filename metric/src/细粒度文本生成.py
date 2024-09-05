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
from transformers import AutoTokenizer,GenerationConfig
from transformers import AutoModelForSeq2SeqLM


device = "cuda:0"
base_file = "./data/train_images.csv"
# tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
# model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
# classifier = pipeline("summarization",min_length=1,max_length=10,do_sample=True)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",device=device)
# summarizer = summarizer.to(device)

output = []
with open(base_file, 'r+') as f:
    data = csv.DictReader(f)
    data = [r for r in data]
    pbar = tqdm(total=len(data))
    for row in data:
        prompt = row["prompt"]
        # Coarse-grained text
        # Fine-grained text
        row["fine-grained text"] = prompt
        __ = []
        ##------
        # 1
        outputs = summarizer(prompt,max_length=5, min_length=1)
        for _ in outputs:
            __.append({"prompt":_["summary_text"]})
        # 2
        outputs = summarizer(prompt,max_length=10, min_length=6)
        for _ in outputs:
            __.append({"prompt":_["summary_text"]})
        # 3
        outputs = summarizer(prompt,max_length=15, min_length=11)
        for _ in outputs:
            __.append({"prompt":_["summary_text"]})

        ##------
        row["coarse-grained prompt"] = __
        # print(output)
        # break
        del row["prompt"]
        output.append(row)
        pbar.update(1)
    pbar.close()
    
with open(f"./data/train_images_coarse-grained.csv.csv", 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)
    

