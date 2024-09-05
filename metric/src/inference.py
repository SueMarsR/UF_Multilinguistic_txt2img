import torch
import os
import csv

from tqdm import tqdm
import hashlib
import shutil
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

data_type = "train"
prompt = "a woman sitting on a beach chair wearing a colorful dress and sunglasses, high detailed skin, skin pores, 8k uhd, high quality, freckles, Fujifilm XT3"
base_file = f"./data/{data_type}.csv"

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, cache_dir="./cache")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
g = torch.Generator(device='cuda')


def rm_mk_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def md5(img):
    md5hash = hashlib.md5(img.tobytes()).hexdigest()
    return md5hash


def get_pipeline_embeds(pipe, prompt, negative_prompt="", device="cuda:0"):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
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
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    return prompt_embeds,negative_prompt_embeds

def pipe_rum(prompt, height = 512, width = 512, step=50, cfg=7.5):
    dic = {}
    seed = g.seed()
    prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt)
    image = pipe(
        # prompt,
        prompt_embeds=prompt_embeds, 
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        generator = g,
        num_inference_steps=step,
        guidance_scale=cfg,
        ).images[0]
    img_name = md5(image)

    image.save(f"images/{data_type}/{img_name}.png")
    dic["prompt"] = prompt
    dic["step"] = step
    dic["seed"] = seed
    dic["height"] = height
    dic["width"] = width
    dic["cfg"] = cfg
    dic["sampler"] = "Euler a"



    # print(seed)
    return dic
if __name__ == "__main__": 
    rm_mk_dir(f"images/{data_type}")

    output = []
    with open(base_file, 'r+') as f:
        data = csv.DictReader(f)
        data = [r for r in data]
    pbar = tqdm(total=len(data))
    for row in data:
        prompt = row["Prompt"]
        dic = pipe_rum(prompt)
        output.append(dic)
        # break
        pbar.update(1)
    pbar.close()
    print(output)

    with open(f"data/{data_type}_images.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)




    # image = pipe(prompt, generator = g).images[0]

    # image.show()
