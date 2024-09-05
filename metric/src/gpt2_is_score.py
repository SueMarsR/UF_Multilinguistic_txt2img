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
import torch
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import timm
from torch.nn import functional as F
from scipy.stats import entropy
import sys

device = "cuda:0"
base_file = "./model_output_pic/gpt2.csv"
gpt_num = "0"


def inception_score(data, splits=10):
    N = len(data)
    preds = np.zeros((len(data), 1000))
    inception_model = timm.create_model("inception_v4", pretrained=True)
    inception_model.to(device)
    inception_model.eval()
    config = resolve_data_config({}, model=inception_model)
    transform = create_transform(**config)
    pbar = tqdm(total=len(data))

    for _, row in enumerate(data):
        image_id = str(row["id"]).zfill(6)
        # for i in range(5):
        prompt = row[f"prompt_{gpt_num}"]
        image = Image.open(f"./model_output_pic/gpt2_{gpt_num}/{image_id}.png")
        # out = inception_model(image)
        image = transform(image).unsqueeze(0).to(device)
        # print(image.shape)
        __ = inception_model(image)
        __ = F.softmax(__).data.cpu().numpy()

        preds[_] = __
        pbar.update(1)
    pbar.close()

    # preds = torch.cat(preds, dim=0)
    # realism_score = torch.mean(torch.max(preds, dim=1)[0].log())
    # diversity_score = torch.exp(torch.mean(torch.sum(preds * torch.log(preds), dim=1)))
    # inception_score = torch.exp(realism_score) * diversity_score
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    output = []
    gpu_id = int(sys.argv[1])
    device = f"cuda:{gpu_id}"
    gpt_num = f"{gpu_id}"
    with open(base_file, "r+") as f:
        data = csv.DictReader(f)
        data = [r for r in data]
        # pbar = tqdm(total=len(data))
        print(inception_score(data))
        # print(is_score)

        # pbar.close()
