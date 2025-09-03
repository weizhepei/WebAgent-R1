from trl import GRPOConfig
from typing import List, Optional

# import os
# os.environ["WANDB_MODE"] = "disabled"

def get_default_grpo_config(run_name: str,
                            num_gpus: int = 1,
                            reward_weights: Optional[List[float]] = None) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"/tmp/instance_storage/browser_r1/outputs/webrl_llama3.1-8b/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant", # [constant_with_warmup, constant]
        warmup_steps=0, # 10% (default: 0)
        num_train_epochs=1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.01,
        num_iterations=1,
        beta=0.001, # 0.001 in verl; 0.04 in open-r1
        max_prompt_length=16382, #8192 * 2 = 16382
        max_completion_length=1024,
        per_device_train_batch_size=1, # 16
        per_device_eval_batch_size=1, # 32
        num_generations=num_gpus-1,
        gradient_accumulation_steps=16, #  int(16 / num_gpus) total_batch_size = per_device_train_batch_size * gradient_accumulation_steps
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=0.125, # save every 0.125 epoch
        save_only_model=True,
        eval_strategy="no", # steps
        eval_steps=0.125,
        eval_on_start = False, # True
        use_vllm=True,
        vllm_device=f"cuda:{num_gpus-1}",
        vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        reward_weights=reward_weights,
        temperature=1.0
    )


