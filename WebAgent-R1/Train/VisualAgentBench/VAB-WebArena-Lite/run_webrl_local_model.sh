#!/bin/bash

source /workspace/setup_vars.sh

# for Chat models
# python run.py \
#   --mode chat \
#   --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
#   --test_start_idx 0 \
#   --test_end_idx 1 \
#   --temperature 1 \
#   --result_dir results_debug \
#   --provider openai \
#   --model Qwen/Qwen2.5-0.5B-Instruct \
#   --planner_ip http://localhost:8000/v1 \
#   --stop_token "<|eot_id|>" \
#   --max_obs_length 0 \
#   --max_tokens 2048 \
#   --viewport_width 1280 \
#   --viewport_height 720 \
#   --action_set_tag webrl_id \
#   --observation_type webrl \
#   --test_config_base_dir config_files/wa/test_webarena_lite
  
# for WebRL/Completion/Non-Chat models
python run.py \
  --mode completion \
  --instruction_path agent/prompts/jsons/p_webrl.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --temperature 1 \
  --result_dir results_debug \
  --provider openai \
  --model THUDM/webrl-llama-3.1-8b \
  --planner_ip http://localhost:8000/v1 \
  --stop_token "<|eot_id|>" \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --test_config_base_dir config_files/wa/test_webarena_lite
  