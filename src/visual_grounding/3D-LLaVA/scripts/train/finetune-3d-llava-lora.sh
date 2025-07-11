cd /inspire/hdd/global_user/xieyuan-24039/sjm/3D-llava-0705
source /opt/conda/etc/profile.d/conda.sh
conda activate llava
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_MODE=disabled

# referring seg
export scanrefer=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/scanrefer_train_3d_llava.json
export multi3drefer=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/multi3drefer_train_3d_llava.json
export nr3d=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/nr3d_train_3d_llava.json

# dense captioning
export scan2cap=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/scan2cap_train_3d_llava.json
export nr3d_caption=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/nr3d_caption_train_3d_llava.json

# vqa
export scanqa=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/scanqa_train_3d_llava.json
export sqa3d=/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/sqa3d_train_3d_llava.json

EXP_NAME=finetune-3d-llava-lora-0708-GMM1CL01COT0-margin-1.0-V2-bs-2-gpu8

CODE_DIR=record/${EXP_NAME}
mkdir -p "$CODE_DIR"
cp -r llava "$CODE_DIR"
    # --data_path $scan2cap $scanqa $sqa3d $nr3d_caption $scanrefer $scanrefer $scanrefer $multi3drefer $nr3d \

PYTHONPATH=$(pwd) \
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --deepspeed /inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA/scripts/zero1_3d_llava.json \
    --model_name_or_path /inspire/hdd/global_user/xieyuan-24039/sjm/LLaVA-3D-bf/llava-v1.5-7b \
    --version v1 \
    --data_path $scan2cap $scanqa $sqa3d $nr3d_caption $scanrefer $scanrefer $scanrefer $multi3drefer $nr3d \
    --scan_folder /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pointcloud_tower /inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA/checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth \
    --pc_modules_to_finetune alignment_proj hidden_seg_fc \
    --num_pc_tokens 100 \
    --inst_prompt_encoder shared_projector \
    --freeze_pointcloud_tower True \
    --pc_use_link_token False \
    --image_aspect_ratio pad \
    --group_by_task_length_per_batch True \
    --bf16 True \
    --output_dir ./checkpoints/${EXP_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /inspire/hdd/global_user/xieyuan-24039/sjm/resnet101.py
bash /inspire/hdd/global_user/xieyuan-24039/sjm/3D-llava-0705/scripts/eval/multigpu_eval_scanrefer.sh