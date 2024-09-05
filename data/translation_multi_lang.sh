export CUDA_VISIBLE_DEVICES=6,7

python -u -m accelerate.commands.launch --num_processes=2 --main_process_port=51115 images_multi_gpu/data/translation_accelerator.py