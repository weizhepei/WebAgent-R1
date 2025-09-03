from typing import Callable, Optional, Union, Any, List

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
import numpy as np
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig # type: ignore

if is_wandb_available():
    import wandb

import json

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env

    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # prompts = [x["prompt"] for x in inputs] # type: ignore
        prompts = inputs # for WebArena

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        
        # print(f'\n>>> prompts_text at device {device} (length: {len(prompts_text)}): {json.dumps(prompts_text, indent=2)}')

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
    
        # prompts_text = gather_object(prompts_text)
        # print(f'\n>>> (after gather) prompts_text at device {device} (length: {len(prompts_text)}): {json.dumps(prompts_text, indent=2)}')

        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        all_inputs = gather_object(inputs)
        if self.accelerator.is_main_process:
            # print(f'\n>>> type(all_prompts) at device {device}: {type(all_prompts)}') # list
            # print(f'length(all_prompts) at device {device}: {len(all_prompts)}') 
            # print(f'type(all_prompts[0]) at device {device}: {type(all_prompts[0])}') # list
            # print(f'length(all_prompts[0]) at device {device}: {len(all_prompts[0])}\n\n') # a 8-turn conversation [{'role': 'system', 'content': 'system prompt'}, {'role': 'user', 'content': 'user prompt'}, {'role': 'assistant', 'content': 'assistant response'}, ...]
            # print(f'all_prompts[0] at device {device}: {all_prompts[0]}') # a multi-turn conversation

            # print(f'\n>>> type(inputs) at device {device}: {type(all_inputs)}')  
            # print(f'length(inputs) at device {device}: {len(all_inputs)}') # length qual to n_genetaions, each is a task_config

            env_result = self.env.generate(
                prompts=all_prompts,
                task_configs=all_inputs, # for WebArena only
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_mask = env_result['mask']

            #DEBUG test max OOM len
            # completion_ids = [i * 5 for i in env_result['ids']]
            # completion_mask = [i * 5 for i in env_result['ids']]

            completion_messages = env_result['messages']
            completion_reward = env_result['reward']
            completion_steps = env_result['steps']

            if self.max_prompt_length is not None:
                completion_ids = [row[-self.max_prompt_length:] for row in completion_ids]
                completion_mask = [row[-self.max_prompt_length:] for row in completion_mask]

            # print(f'\n>>> completion_ids len (is_main_process: {self.accelerator.is_main_process} at device {device}): {len(completion_ids)}, shape: {[len(i) if i is not None else None for i in completion_ids]}')
            # print(f'\n>>> completion_messages len (is_main_process: {self.accelerator.is_main_process} at device {device}): {len(completion_messages)}, shape: {[len(i) if i is not None else None for i in completion_messages]}')
            # print(f'\n>>> completion_mask len (is_main_process: {self.accelerator.is_main_process} at device {device}): {len(completion_mask)}, shape: {[len(i) if i is not None else None for i in completion_mask]}')
            # print(f'\n>>> completion_reward len (is_main_process: {self.accelerator.is_main_process} at device {device}): {len(completion_reward)},{[i if i is not None else None for i in completion_reward]}')

        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            completion_reward = [None] * len(all_prompts)
            completion_steps = [None] * len(all_prompts)
        
            # print(f'\n>>> completion_ids len (is_main_process: {self.accelerator.is_main_process}): {len(completion_ids)}, shape: {[i.shape if i is not None else None for i in completion_ids]}')
            # print(f'\n>>> completion_messages len (is_main_process: {self.accelerator.is_main_process}): {len(completion_messages)}, shape: {[i.shape if i is not None else None for i in completion_messages]}')
            # print(f'\n>>> completion_mask len (is_main_process: {self.accelerator.is_main_process}): {len(completion_mask)}, shape: {[i.shape if i is not None else None for i in completion_mask]}')

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        # completion_page = broadcast_object_list(completion_page, from_process=0)
        # completion_env_ids = broadcast_object_list(completion_env_ids, from_process=0)
        completion_reward = broadcast_object_list(completion_reward, from_process=0)
        completion_steps = broadcast_object_list(completion_steps, from_process=0)

        # print(f'\n>>> [after broadcast] completion_ids len (is_main_process: {self.accelerator.is_main_process}): {len(completion_ids)}, shape: {[len(i) if i is not None else None for i in completion_ids]}')
        # print(f'\n>>> [after broadcast] completion_messages len (is_main_process: {self.accelerator.is_main_process}): {len(completion_messages)}, shape: {[len(i) if i is not None else None for i in completion_messages]}')
        # print(f'\n>>> [after broadcast] completion_mask len (is_main_process: {self.accelerator.is_main_process}): {len(completion_mask)}, shape: {[len(i) if i is not None else None for i in completion_mask]}')

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        # completion_page = completion_page[process_slice]
        # completion_env_ids = completion_env_ids[process_slice]
        completion_reward = completion_reward[process_slice]
        completion_steps = completion_steps[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)

        if self.max_prompt_length is not None:
            prompt_completion_ids = prompt_completion_ids[:, -self.max_prompt_length:]
            attention_mask = attention_mask[:, -self.max_prompt_length:]

        logits_to_keep = completion_ids.size(1)

        # print(f'>>> Starting to compute rewards')
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                # print(f'\nprompt_completion_ids shape: {prompt_completion_ids.shape}')
                # print(f'\nattention_mask shape: {attention_mask.shape}')
                # print(f'\nlogits_to_keep: {logits_to_keep}')
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None
            # print(f'\nFinished computing old_per_token_logps')

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        
        # print(f'>>> Finished computing rewards')

        # use message dicts for reward function inputs
        completions = completion_messages
        llm_as_judge_reward = completion_reward
        
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # print(f'\n>>> init rewards_per_func (is_main_process: {self.accelerator.is_main_process} at device {device}): shape {rewards_per_func.shape}\n{rewards_per_func}')
        # print(f'\n>>> completion_reward (is_main_process: {self.accelerator.is_main_process} at device {device}): length {len(completion_reward)}\n{completion_reward}')

        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, llm_as_judge_reward=llm_as_judge_reward, **reward_kwargs) # type: ignore
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device) # check the shape of rewards_per_func

        rewards_per_func = gather(rewards_per_func) # check the shape of rewards_per_func

        # print(f'\n>>> (after gather) rewards_per_func (is_main_process: {self.accelerator.is_main_process} at device {device}):shape {rewards_per_func.shape}\n{rewards_per_func}')

        # print(f'\n >>> Finished gathering rewards')
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # print(f'Starting to slice the data')
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        # print(f'\n>>> completion_mask shape: {completion_mask.shape}')
        # print(f'\n>>> completion_steps: {completion_steps} with type: {type(completion_steps)}') # type is list

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        completion_steps_gathered = self.accelerator.gather_for_metrics(completion_steps)
        # print(f'\n>>> (after metric gather) completion_steps_gathered: {completion_steps_gathered} with type: {type(completion_steps_gathered)}')
        # print(f'\n>>> (after metric gather) completion_length: {completion_length} with shape: {completion_length.shape}')

        completion_length = completion_length.float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["n_rounds"].append(np.mean(completion_steps_gathered))
        # self._metrics[mode]["completion_steps"].append(completion_steps)

        # print(f'>>> (after metric gather) completion_steps: {completion_steps} with type: {type(completion_steps)}')

        reward_per_func = rewards_per_func.mean(0) # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # print(f'>>> Finished logging rewards')

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts = [x["prompt"] for x in inputs] # only for webarena
            prompts_to_log = gather_object(prompts)

            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    # idx = 0

                    # find the max reward index
                    idx = rewards.argmax().item()

                    # find the index with shortest completion_message, completion_message is a list of list
                    # idx = min(
                    #     range(len(completion_messages)),
                    #     key=lambda i: len(completion_messages[i])
                    # )

                    print_prompt_completions_sample(
                        [str(prompts_to_log[idx][-1]["content"])],
                        [completions_to_log[idx]],
                        [rewards_to_log[idx]],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }