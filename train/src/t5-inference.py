from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from models import MLP
import torch
import os
from dataset.utils import pre_caption
from transformers import pipeline

CUDA_VISIBLE_DEVICES=2

text_encoder_checkpoint = "/data/mty/UF-FGTG_code/train/work_dir/sd_diffuser_gpt_test_ZH_5e-5/pipeline-final"
text_decoder_checkpoint = "google-t5/t5-base"
# text_decoder_checkpoint = "/data/mty/UF-FGTG_code/train/work_dir/sd_diffuser_gpt_test_ZH_5e-5/pipeline-final"


# prompt = "a portrait of a girl skull face,"
# prompt_coarse_ZH = "一个漫画画像，描绘了一个赛博朋克的机械少女"
# prompt_coarse_original = "a comic potrait of a cyberpunk cyborg girl with"
# prompt_fine = "a comic potrait of a cyberpunk cyborg girl with big and cute eyes, fine - face, realistic shaded perfect face, fine details. night setting. very anime style. realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, trending on art station"

prompt_coarse_ZH = "娜塔莉·波特曼(Natalie Portman)是乐观的, 充满欢乐的, 昏暗的中世纪旅馆老板."
prompt_coarse_original = "Natalie Portman as optimistic!, cheerful, giddy medieval innkeeper in dark shadows ."
prompt_fine = "young, curly haired, redhead Natalie Portman  as a optimistic!, cheerful, giddy medieval innkeeper in a dark medieval inn. dark shadows, colorful, candle light,  law contrasts, fantasy concept art by Jakub Rozalski, Jan Matejko, and J.Dickenson"


base_diffuser_model = "stabilityai/stable-diffusion-2-1-base"
weight_dtype = torch.float32
torch.set_default_dtype(weight_dtype)
device = "cuda:0"

if __name__ == "__main__":
    ## encoder CLIP
    print(prompt_coarse_ZH)
    prompt = pre_caption(prompt_coarse_ZH, 256)
    print(prompt)
    # text_encoder = CLIPTextModel.from_pretrained(os.path.join(text_encoder_checkpoint, "diffusion"), subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(text_encoder_checkpoint, "text_encoder"), torch_dtype=weight_dtype)
    text_encoder_tokenizer = CLIPTokenizer.from_pretrained(base_diffuser_model, subfolder="tokenizer", cache_dir="./cache")

    ## adapter
    text_adapter = MLP(dtype="fp32", input_dim=1024, output_dim=768)
    text_adapter = torch.nn.DataParallel(text_adapter, device_ids=[0])
    text_adapter.load_state_dict(torch.load(os.path.join(text_encoder_checkpoint, "text_adapter", "text_adapter.bin")))

    ## decoder T5
    text_decoder = T5ForConditionalGeneration.from_pretrained(text_decoder_checkpoint, cache_dir="./cache", torch_dtype=weight_dtype)
    # text_decoder = T5ForConditionalGeneration.from_pretrained(os.path.join(text_decoder_checkpoint, "text_decoder"), torch_dtype=weight_dtype)
    text_decoder_tokenizer = AutoTokenizer.from_pretrained(text_decoder_checkpoint, cache_dir="./cache")
    # text_decoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(text_decoder_checkpoint, "text_decoder"))

    ##
    text_encoder = text_encoder.to(device)
    # text_encoder_tokenizer = text_encoder_tokenizer.to(device)
    text_adapter = text_adapter.to(device)
    text_decoder = text_decoder.to(device)
    # text_decoder_tokenizer = text_decoder_tokenizer.to(device)

    inputs = text_encoder_tokenizer(prompt, max_length=text_encoder_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    inputs = inputs.to(device)
    inputs = text_encoder(inputs)[0]
    inputs = text_adapter(inputs)
    
    translation_generation_config = GenerationConfig(
        num_beams=5,
        num_beam_groups=5,
        max_new_tokens=32,
        num_return_sequences=5,
        output_scores=True,
        diversity_penalty=0.5,
        # do_sample=True,
    )

    outputs = text_decoder.generate(
        inputs_embeds=inputs,
        generation_config=translation_generation_config,
    )

    output = text_decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    output_prompt = ""
    for _ in output:
        output_prompt += _
    print(output_prompt)
    
    # model_name = "/data/mty/UF-FGTG_code/images_multi_gpu/cache_model/models--Falconsai--text_summarization/snapshots/6e505f907968c4a9360773ff57885cdc6dca4bfd"
    # summarizer = pipeline("summarization", model=model_name)

    # summarization = summarizer(output_prompt, max_length=24, min_length=8, do_sample=False)
    # print(summarization[0]['summary_text'])

