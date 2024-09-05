import torch
import os
import csv

from tqdm import tqdm
import hashlib
import shutil
import sys
import math
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import ast

# data_type = "eval"
# prompt = "a woman sitting on a beach chair wearing a colorful dress and sunglasses, high detailed skin, skin pores, 8k uhd, high quality, freckles, Fujifilm XT3"
# base_file = "./data/eval_images_coarse-grained_ans_5.csv"
base_file = "./data_comparison/eval_images_coarse-graied_promptist.csv"
model_type = "promptist"


def rm_mk_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def md5(img):
    md5hash = hashlib.md5(img.tobytes()).hexdigest()
    return md5hash


def get_pipeline_embeds(pipe, prompt, device, negative_prompt=""):
    """Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipe.tokenizer.model_max_length

    input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
    negative_ids = negative_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i : i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i : i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    return prompt_embeds, negative_prompt_embeds


def pipe_rum(id, raw_data, pipe, g, device, height=512, width=512, step=50, cfg=7.5):
    dic = {}
    dic["id"] = id

    for idx in range(0, 1):
        prompt = ast.literal_eval(raw_data[f"ans_{idx}"])["prompt"].decode("utf-8")
        # print(prompt)
        dic[f"prompt_{idx}"] = prompt
        seed = g.seed()
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, device=device)
        image = pipe(
            # prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
            generator=g,
            num_inference_steps=step,
            guidance_scale=cfg,
        ).images[0]
        img_name = str(id).zfill(6)

        output = f"model_output_pic/{model_type}_{idx}/{img_name}.png"
        image.save(output)
    # print(seed)
    return dic


if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    device = f"cuda:{gpu_id}"
    model_type = sys.argv[2]
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, cache_dir="./cache")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)

    g = torch.Generator(device=device)

    # for idx in range(0, 5):
    #     rm_mk_dir(f"./model_output_pic/{model_type}_{idx}")

    output = []
    with open(base_file, "r+") as f:
        data = csv.DictReader(f)
        data = [r for r in data]
    pbar = tqdm(total=math.ceil(len(data) / 8))
    for i, row in enumerate(data):
        if i % 8 == gpu_id:
            dic = pipe_rum(i, row, pipe=pipe, g=g, device=device)
            output.append(dic)
            # break
            pbar.update(1)
    pbar.close()
    # print(output)

    with open(f"model_output_pic/{model_type}_multi_gpu_{gpu_id}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    # image = pipe(prompt, generator = g).images[0]

    # image.show()
