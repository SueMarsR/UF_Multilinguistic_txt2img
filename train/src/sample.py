import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, ChineseCLIPConfig
from models import MLP
import os
from dataset.utils import pre_caption
import ast

CUDA_VISIBLE_DEVICES = 1

# prompt_coarse_ZH = "一个漫画画像，描绘了一个赛博朋克的机械少女"
# prompt_coarse_original = "a comic potrait of a cyberpunk cyborg girl with"
# prompt_fine = "a comic potrait of a cyberpunk cyborg girl with big and cute eyes, fine - face, realistic shaded perfect face, fine details. night setting. very anime style. realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, trending on art station"

# prompt_coarse_ZH = "黑色的高桥涼介的美感插图"
# prompt_coarse_original = "aesthetic illustration of ryosuke takahashi with black"
# prompt_fine = "aesthetic illustration of ryosuke takahashi with black hair, dark blue shirt and white pants, standing by his white glossy 1990 mazda rx-7fc3s on an empty highway at sunrise, cinematic lighting, initial d anime 1080 p, 90 s anime aesthetic, volumetric lights, rule of thirds, unreal engine 5 render, pinterest wallpaper, trending on artstation"

# prompt_coarse_ZH = "娜塔莉·波特曼(Natalie Portman)是乐观的, 充满欢乐的, 昏暗的中世纪旅馆老板."
# prompt_coarse_original = "Natalie Portman as optimistic!, cheerful, giddy medieval innkeeper in dark shadows ."
# prompt_fine = "young, curly haired, redhead Natalie Portman  as a optimistic!, cheerful, giddy medieval innkeeper in a dark medieval inn. dark shadows, colorful, candle light,  law contrasts, fantasy concept art by Jakub Rozalski, Jan Matejko, and J.Dickenson"
# negative_prompt = "bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs,"

prompt_coarse_ZH = "绵长蜿蜒的海滩，热带风情，明亮，简单"
prompt_coarse_original = " A long and winding beach, tropical, bright, simple"
prompt_fine = "A long and winding beach, tropical, bright, simple, by Studio Ghibli and Greg Rutkowski, artstation",

text_encoder_checkpoint = "/data/mty/UF-FGTG_code/train/work_dir/sd_diffuser_gpt_test_ZH_5e-5/pipeline-final"
text_decoder_checkpoint = "google-t5/t5-base"

base_diffuser_model = "/data/mty/UF-FGTG_code/train/cache/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
weight_dtype = torch.float32
torch.set_default_dtype(weight_dtype)
device = "cuda:0"
# checkpoint_diffuser_model = "/data/mty/UF-FGTG_code/train/work_dir/sd_diffuser_gpt_test_ZH5e-5/pipeline-35/diffusion"

## encoder CLIP
# print(prompt_coarse_ZH)
prompt = pre_caption(prompt_coarse_ZH, 256)
prompt_coarse_original = pre_caption(prompt_coarse_original, 256)
print(prompt)


# ## Load Text Encoder, Adapter, and Decoder

# # text_encoder = CLIPTextModel.from_pretrained(os.path.join(text_encoder_checkpoint, "diffusion"), subfolder="text_encoder", torch_dtype=weight_dtype)
# text_encoder = CLIPTextModel.from_pretrained(os.path.join(text_encoder_checkpoint, "text_encoder"), torch_dtype=weight_dtype)
# text_encoder_tokenizer = CLIPTokenizer.from_pretrained(base_diffuser_model, subfolder="tokenizer", cache_dir="./cache")

# ## adapter
# text_adapter = MLP(dtype="fp32", input_dim=1024, output_dim=768)
# text_adapter = torch.nn.DataParallel(text_adapter, device_ids=[0])
# text_adapter.load_state_dict(torch.load(os.path.join(text_encoder_checkpoint, "text_adapter", "text_adapter.bin")))

# ## decoder T5
# text_decoder = T5ForConditionalGeneration.from_pretrained(text_decoder_checkpoint, cache_dir="./cache", torch_dtype=weight_dtype)
# # text_decoder = T5ForConditionalGeneration.from_pretrained(os.path.join(text_decoder_checkpoint, "text_decoder"), torch_dtype=weight_dtype)
# text_decoder_tokenizer = AutoTokenizer.from_pretrained(text_decoder_checkpoint, cache_dir="./cache")
# # text_decoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(text_decoder_checkpoint, "text_decoder"))

# ##
# text_encoder = text_encoder.to(device)
# # text_encoder_tokenizer = text_encoder_tokenizer.to(device)
# text_adapter = text_adapter.to(device)
# text_decoder = text_decoder.to(device)
# # text_decoder_tokenizer = text_decoder_tokenizer.to(device)

# translation_generation_config = GenerationConfig(
#     num_beams=5,
#     num_beam_groups=5,
#     max_new_tokens=32,
#     num_return_sequences=3,
#     output_scores=True,
#     diversity_penalty=0.5,
#     # do_sample=True,
# )

# inputs = text_encoder_tokenizer(prompt, max_length=text_encoder_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
# inputs = inputs.to(device)
# inputs = text_encoder(inputs)[0]
# inputs = text_adapter(inputs)
# # outputs = text_decoder.generate(inputs_embeds=inputs)
# # print(text_decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True))

# outputs = text_decoder.generate(
#     inputs_embeds=inputs,
#     generation_config=translation_generation_config,
# )
# # # print(outputs.scores)
# # print(outputs, outputs.shape, outputs.dtype)
# # # output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# output = text_decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)

# output_prompt = ""
# for _ in output:
#     output_prompt += _
# print(output_prompt)
# # for _ in output:
# #     print(_)
        
## Image Generation

def get_pipeline_embeds(tokenizer, text_encoder, prompt, device, negative_prompt=""):
    """Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = tokenizer.model_max_length

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    negative_ids = tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
    negative_ids = negative_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(text_encoder(input_ids[:, i : i + max_length])[0])
        neg_embeds.append(text_encoder(negative_ids[:, i : i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    return prompt_embeds, negative_prompt_embeds

def pipe_rum(id, prompt, pipe, tokenizer, text_encoder, g, device, negative_prompt, height=512, width=512, step=50, cfg=7.5):
    dic = {}
    dic["id"] = id

    # for idx in range(0, 1):
    # prompt = ast.literal_eval(raw_data[f"ans_{idx}"])["prompt"].decode("utf-8")
    # print(prompt)
    # dic[f"prompt_{idx}"] = prompt
    seed = g.seed()
    prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe,tokenizer, text_encoder, prompt, device, negative_prompt)
    image = pipe(
        # prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        generator=g,
        num_inference_steps=step,
        guidance_scale=cfg,
        seed=seed,
    ).images[0]
    img_name = str(id).zfill(6)

    # output_path = f"/data/mty/UF-FGTG_code/data_analysis/output/sample_{id}_OURS_{img_name}.png"
    # image.save(output_path)
    
    # print(seed)
    return image

pipe = StableDiffusionPipeline.from_pretrained(base_diffuser_model, torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
# print(pipe)

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", cache_dir="./cache")
# print("ChineseCLIP: ", model)
text_encoder = model.text_model
text_encoder = text_encoder.to(device)
# print("ChineseCLIP: ", text_encoder)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", cache_dir="./cache")


## OURS
g = torch.Generator(device=device)
seed = g.seed()
image_ours = pipe_rum(2, tokenizer=processor, prompt=prompt, pipe=pipe, text_encoder=text_encoder, g=g, device=device, negative_prompt="", height=512, width=512, step=50, cfg=7.5)
image_ours.show()

## SD
# image_SD = pipe(prompt, generator = g).images[0]
image_SD = pipe(prompt, height=512,width=512,generator = g).images[0]
# image.save(f"/data/mty/UF-FGTG_code/data_analysis/output/sample_2_SD_fine.png")
image_SD.show()


