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
device="cuda:0"

model = "michellejieli/NSFW_text_classifier"
# model = "michellejieli/inappropriate_text_classifier"
classifier = pipeline("sentiment-analysis", model=model,device=device)
base_file = "./data/eval_images_coarse-grained.csv"

if __name__ == "__main__":
    output = []
    cnt = 0
    with open(base_file, 'r+') as f:
        data = csv.DictReader(f)
        data = [r for r in data]
        pbar = tqdm(total=len(data))
        for row in data:
            prompt = row["fine-grained prompt"]
            _ = classifier(prompt)[0]
            if _["label"] == "NSFW":
                if _["score"] < 0.9:
                    _["label"] = "SFW"
                    _["score"] = 1 - _["score"]
                
            row["NSFW_label"] = _["label"]
            row["NSFW_score"] = _["score"]
            if _["label"] != "NSFW":
                cnt += 1
            output.append(row)
            pbar.update(1)
        pbar.close()
    print(len(data),cnt)
        
    with open(f"./data/eval_images_coarse-grained_NSFW.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)