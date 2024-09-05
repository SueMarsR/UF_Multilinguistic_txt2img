import torch
import torch.nn as nn
import sys 
sys.path.insert(0, sys.path[0]+"/../")
from dataset.utils import pre_caption

torch.cuda.empty_cache()

CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1

prompt_coarse_ZH = "绵长蜿蜒的海滩，热带风情，明亮，简单"
prompt_coarse_original = " A long and winding beach, tropical, bright, simple"
prompt_fine = "A long and winding beach, tropical, bright, simple, by Studio Ghibli and Greg Rutkowski, artstation",

prompt = pre_caption(prompt_coarse_ZH, 256)
prompt_coarse_original = pre_caption(prompt_coarse_original, 256)

device = "cuda"
g = torch.Generator(device=device)
seed = g.seed()

# LLAMA2-Chinese

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "/data/mty/UF-FGTG_code/train/cache/models/models--LinkSoul--Chinese-Llama-2-7b/snapshots/72efd71d7f89d9c46008b7a574faf90300ed9ba8"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# print(tokenizer.model_max_length) # 4096
# print("LLaMA2 Model: ", model)
# print("Streamer: ", streamer)

instruction = """[INST] <<SYS>>\n
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
            Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            Please expand the following text in English so that it applies to the generated images. \n<</SYS>>\n
            \n{} [/INST]"""


prompt = instruction.format("绵长蜿蜒的海滩")
# prompt = instruction.format("一个漫画画像，描绘了一个赛博朋克的机械少女")

print(prompt)

input_LC = tokenizer(prompt, return_tensors='pt')
input_ids_LS = input_LC.input_ids.cuda()
# print(input_ids_LS.shape)

# generate_ids = model.generate(input_ids_LS, max_new_tokens=4096, streamer=streamer)
# print(generate_ids.shape)

embedding_LC = model(input_ids_LS)
print(embedding_LC)
print(embedding_LC.logits.shape)

linear = nn.Linear(32000, 768).to(device)

# Stable Diffusion
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

base_diffuser_model = "/data/mty/UF-FGTG_code/train/cache/models/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
pipe = StableDiffusionPipeline.from_pretrained(base_diffuser_model, torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)



# # anything-midjourney

# from diffusers import DiffusionPipeline

# model_path_mid="/data/mty/UF-FGTG_code/train/cache/models/models--stablediffusionapi--anything-midjourney/snapshots/77716cc9746e86a5aff527860d228e1a7085c77c"

# pipeline_mid = DiffusionPipeline.from_pretrained(model_path_mid).to(device)

# image_mid = pipeline_mid(prompt, generator=g, height=512, width=512).images[0]
# image_mid.show()

# # print(pipeline)