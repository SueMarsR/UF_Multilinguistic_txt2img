import torch
import os
import csv

from tqdm import tqdm
import hashlib
import shutil
import sys
import math
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

data_type = "eval"
prompt = "a woman sitting on a beach chair wearing a colorful dress and sunglasses, high detailed skin, skin pores, 8k uhd, high quality, freckles, Fujifilm XT3"
base_file = f"./data/{data_type}.csv"


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


def pipe_rum(prompt, pipe, g, device, height=512, width=512, step=50, cfg=7.5):
    dic = {}
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
    img_name = md5(image)
    output = f"images_multi_gpu/{data_type}/{img_name}.png"
    image.save(output)
    dic["prompt"] = prompt
    dic["step"] = step
    dic["seed"] = seed
    dic["height"] = height
    dic["width"] = width
    dic["cfg"] = cfg
    dic["sampler"] = "Euler a"
    dic["image_name"] = f"{img_name}.png"
    dic["image_path"] = output.split("/", 1)[-1]

    # print(seed)
    return dic


if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    device = f"cuda:{gpu_id}"
    data_type = sys.argv[2]
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, cache_dir="./cache")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)

    g = torch.Generator(device=device)

    # rm_mk_dir(f"images/{data_type}")

    output = []
    with open(base_file, "r+") as f:
        data = csv.DictReader(f)
        data = [r for r in data]
    pbar = tqdm(total=math.ceil(len(data) / 8))
    for i, row in enumerate(data):
        if i % 8 == gpu_id:
            prompt = row["Prompt"]
            dic = pipe_rum(prompt, pipe=pipe, g=g, device=device)
            output.append(dic)
            # break
            pbar.update(1)
    pbar.close()
    # print(output)

    with open(f"data/{data_type}_images_multi_gpu_{gpu_id}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    # image = pipe(prompt, generator = g).images[0]

    # image.show()
