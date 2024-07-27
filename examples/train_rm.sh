python train_reward_model.py --pretrain 'openai-community/gpt2-medium' \
                             --model 'gpt2' \
                             --strategy naive \
                             --loss_fn 'log_exp'\
                             --save_path output_rm \
                             --test True
