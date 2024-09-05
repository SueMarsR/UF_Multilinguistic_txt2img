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
from torchmetrics.functional.multimodal import clip_score
import math
from functools import partial

device = "cuda:0"
base_file = "./data/eval_images.csv"

# gpt_num = "0"


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
        # image_id = str(row["id"]).zfill(6)
        # for i in range(5):
        prompt = row[f"prompt"]
        image_path = os.path.join("images_multi_gpu", row[f"image_path"])
        image = Image.open(image_path)
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


def calculate_clip_score(data, splits=40):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    # clip_score_fn.to(device)
    output = []
    pbar = tqdm(total=len(data))
    prompts = []
    images = []
    for k in range(splits):
        for _, row in enumerate(data[k * (len(data) // splits) : (k + 1) * (len(data) // splits)]):
            # if _ > 100:
            # break
            # prompt = row[f"prompt"]
            prompts.append(row[f"prompt"][:77])
            image_path = os.path.join("images_multi_gpu", row[f"image_path"])
            image = Image.open(image_path)
            image_int = np.asarray(image)
            image_int = np.expand_dims(image_int, 0)
            images.append(image_int)
            pbar.update(1)
        images_int = np.concatenate(images, axis=0)
        with torch.no_grad():
            tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2)
            tensor = tensor.to(device)
            score = clip_score_fn(tensor, prompts).detach()
            tensor.cpu()
            torch.cuda.empty_cache()
        output.append(float(score))
        print(np.mean(output), np.std(output))
    pbar.close()
    # print(float(score))
    # return float(score)
    return np.mean(output), np.std(output)


if __name__ == "__main__":
    output = []
    # gpu_id = int(sys.argv[1])
    # device = f"cuda:{gpu_id}"
    # gpt_num = f"{gpu_id}"
    with open(base_file, "r+") as f:
        data = csv.DictReader(f)
        data = [r for r in data]
        # pbar = tqdm(total=len(data))
        # print(inception_score(data))
        print(calculate_clip_score(data))
        # print(is_score)

        # pbar.close()
