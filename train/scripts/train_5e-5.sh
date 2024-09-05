export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export lr=5e-5
export OUTPUT_DIR="work_dir/sd_diffuser_gpt_test_"$lr
# export dataset_name="lambdalabs/pokemon-blip-captions"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

accelerate launch --mixed_precision="fp16" --main_process_port=51111 train_text_to_image_L.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="/mnt/bn/nlhei-nas/project/diffusion_gpt/Datasets/Stable-Diffusion-Prompts/data/train_images.csv" \
  --images_base_path="/mnt/bn/nlhei-nas/project/diffusion_gpt/Datasets/Stable-Diffusion-Prompts/images_multi_gpu" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=100 \
  --learning_rate=$lr \
  --max_grad_norm=1 \
  --checkpointing_steps=-1 \
  --checkpointing_epochs=5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=1000 \
  --seed=420 \
  --text_loss_gamma=0.1 \
  --output_dir=$OUTPUT_DIR \
  --cache_dir="cache_dir/diffuser_gpt_test" 