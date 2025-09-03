import verifiers as vf
# from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT
from verifiers.utils import get_model_and_tokenizer
from verifiers.envs.webarena_env import WebArenaEnv

from vllm import LLM, SamplingParams
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "THUDM/webrl-llama-3.1-8b"


# model, tokenizer = get_model_and_tokenizer(model_name)

# Initialize tool environment for GSM8K
vf_env = WebArenaEnv(
    # system_prompt="You are a professional web browsing agent assistant that can fulfill user's high-level instructions.",
    system_prompt="",
    dataset="webarena-lite",
    # few_shot=CALCULATOR_FEW_SHOT[0],
    # tools=[calculator],
    max_steps=15,
    n_contexts=3,
)

dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset()
rubric = vf_env.get_rubric()

print(f'>>> dataset size: {len(dataset)}')
print(f'>>> eval_dataset size: {len(eval_dataset)}')

print(f'>>> train example: {json.dumps(dataset[0], indent=4)}')
print(f'\n>>> eval example: {json.dumps(eval_dataset[0], indent=4)}')

# Create a conversation (assuming you want to use the first example from the dataset)
# conversations = [
#     [
#         {"role": "user", "content": f"Task Instruction: {dataset[0]['prompt']}"},
#     ]
# ]

# system_prompt = {'role': 'system', 'content': vf_env.system_prompt}
# # example = CALCULATOR_FEW_SHOT[0]
# example.insert(0, system_prompt)

# conversations = conversations[:1]

# example = []
# conversations = [dataset[0]['prompt']]
# for idx, conv in enumerate(conversations):
#     conversations[idx] = example + conv
#     print(f'\n>>>conversation {idx}:\n{json.dumps(conversations[idx], indent=4)}')


# Initialize vLLM
llm = LLM(model=model_name)

tokenizer = llm.get_tokenizer()

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=1,
    top_p=0.9,
    max_tokens=2048
)

# Generate response using the environment
env_result = vf_env.generate(
    prompts=None,
    llm=llm,
    sampling_params=sampling_params,
    task_configs = [eval_dataset[19], eval_dataset[19], eval_dataset[19], eval_dataset[19]][:3]
)

# Print the result
print("Generated Response:")
print(f'n_responses: {len(env_result["messages"])}')
for idx, response in enumerate(env_result["messages"]):
    print(f'\n>>>Response {idx}:\n{json.dumps(response, indent=4)}')
    print(f'\n>>>Response {idx} ID (length: {len(env_result["ids"][idx])}):\n{env_result["ids"][idx]}: {json.dumps(tokenizer.decode(env_result["ids"][idx]))}')
    print(f'\n>>>Response {idx} Mask (length: {len(env_result["mask"][idx])}):\n{env_result["mask"][idx]}, sum: {sum(env_result["mask"][idx])}')

    # selected ids according to the mask
    selected_ids = [env_result["ids"][idx][i] for i, mask in enumerate(env_result["mask"][idx]) if mask == 1]
    print(f'\n>>>Response {idx} Selected ID (length: {len(selected_ids)}):\n{selected_ids}:\n{tokenizer.convert_ids_to_tokens(selected_ids)}\n{json.dumps(tokenizer.decode(selected_ids))}')
    print(f'\n>>>Response {idx} reward: {env_result["reward"][idx]}')

