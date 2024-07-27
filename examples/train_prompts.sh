# set_n_least_used_CUDA_VISIBLE_DEVICES() {
#     local n=${1:-"9999"}
#     echo "GPU Memory Usage:"
#     local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
#         | tail -n +2 \
#         | nl -v 0 \
#         | tee /dev/tty \
#         | sort -g -k 2 \
#         | awk '{print $1}' \
#         | head -n $n)
#     export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
#     echo "Now CUDA_VISIBLE_DEVICES is set to:"
#     echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# }

# set_n_least_used_CUDA_VISIBLE_DEVICES 1

torchrun --standalone --nproc_per_node=4 train_prompts.py \
--prompt_path samples.json \
--pretrain_dataset InstructionWild/data/instinwild_en.json \
--strategy colossalai_gemini \
--model gpt2 --pretrain "openai-community/gpt2-xl" \
--rm_model gpt2 --rm_pretrain "openai-community/gpt2-medium" \
--train_batch_size 32 --ptx_batch_size 32 --experience_batch_size 32 \
--num_episodes 1 --max_timesteps 5 --update_timesteps 1 --max_epochs 1 \
--lora_rank 128 --ptx_coef 0 \
--cpu_offloading true \
--checkpoint true 

# --strategy naive ddp colossalai_zero2 colossalai_gemini

# --model gpt2 --pretrain "openai-community/gpt2-xl" \
# --rm_model gpt2 --rm_pretrain "openai-community/gpt2-medium" \
# --train_batch_size 32 --ptx_batch_size 32 --experience_batch_size 32 \

# --model opt --pretrain "facebook/opt-1.3b" \
# --rm_model opt --rm_pretrain "facebook/opt-350m" \
# --train_batch_size 32 --ptx_batch_size 32 --experience_batch_size 32 \

# --checkpoint --cpu_offloading