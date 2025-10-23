# script.py
import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT
from verifiers.trainers import GRPOEnvTrainer
from verifiers.envs import ToolEnv
from verifiers.utils import get_default_grpo_config
from verifiers.envs.webarena_env import WebArenaEnv

model_name = "Qwen/Qwen2.5-3B-Instruct"

model, tokenizer = vf.get_model_and_tokenizer(model_name)

TASK = 'webarena-lite'

vf_env = WebArenaEnv(
    dataset=TASK,
    max_steps=15,
    n_contexts=4 # n_gpu - 1
)

dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset()

rubric = vf_env.get_rubric()

training_args = get_default_grpo_config(
    run_name=f"{TASK}_GRPO/Standard/" + model_name.split("/")[-1].lower(),
    num_gpus=5
)

trainer = GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()