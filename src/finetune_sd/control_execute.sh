#!/bin/bash 

export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./../../models"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=./../../data/processed/pokemons/ \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=10000 \
 --checkpointing_steps=2000 \
 --validation_image "./cond_0.png" \
 --validation_prompt "red pokemon with black wings" \
 --train_batch_size=4 \
 --image_column="image" \
 --caption_column="text" \
 --conditioning_image_column='conditioning_image' \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \

git add --all
git commit -m "New results"
git push

poweroff