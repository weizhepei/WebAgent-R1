#!/bin/bash

python run.py \
  --instruction_path agent/prompts/jsons/p_webrl_chat.json \
  --test_start_idx 0 \
  --test_end_idx 2 \
  --result_dir results \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --provider openai \
  --model Qwen/Qwen2.5-32B-Instruct \
  --mode chat \
  --planner_ip '' \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --action_set_tag webrl_id  --observation_type webrl