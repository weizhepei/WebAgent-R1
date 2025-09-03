import inspect
import json
import os
import re
import time
from typing import List, Dict, Sequence, Any, Callable

from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.environment import LLM, SamplingParams
from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers import XMLParser
from verifiers.prompts.system_prompts import WEBARENA_SYS_PROMPT
from verifiers.rubrics.webarena_rubric import WebArenaRubric
from verifiers.utils import preprocess_dataset

from verifiers.envs.WebArena.test_webarena import (
    URL_MAPPINGS,
    get_site_comb_from_filepath,
    prepare, 
    config, 
    generate_trajectory, 
    create_none_action,
    create_webrl_id_based_action,
    evaluator_router,
    tempfile,
    subprocess,
    ScriptBrowserEnv,
    AsyncScriptBrowserEnv,
    StringEvaluator, 
    URLExactEvaluator, 
    HTMLContentExactEvaluator
)

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
    
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any"),
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

def apply_webrl_format(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Apply WebRL format to the messages."""
    formatted_msg = ''
    for idx, msg in enumerate(messages):
        if idx == 0:
            assert msg['role'] == 'user'
            intent, obs = msg['content'].split('Round 0')
            intent, obs = intent.strip(), obs.strip()
            formatted_msg += f'Task Instruction: {intent}\n\nRound 0\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n'
            if obs != '** Simplified html **':
                assert len(messages) == 1
                formatted_msg += obs + '\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
            else:
                formatted_msg += intent + '\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
        elif msg['role'] == 'assistant':
            formatted_msg += msg["content"].strip() + f'\n\nRound {int(idx/2)+1}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n'
        elif msg['role'] == 'user':
            obs = msg["content"].split(f"Round {int(idx/2)}\n\n")[-1].strip()
            formatted_msg += obs + '\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
    
    return [{'role': 'user', 'content': formatted_msg}]

class WebArenaEnv(MultiStepEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 tools: List[Callable] = [],
                 system_prompt: str = WEBARENA_SYS_PROMPT,
                 few_shot: List[Dict[str, str]] = [],
                 mask_env_response: bool = True,
                 max_steps: int = 10,
                 n_contexts=3,
                 **kwargs):
        
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        # self.webarena_path = '/workspace/VisualAgentBench/VAB-WebArena-Lite/'

        self.n_contexts = n_contexts
        self.env = self.init_env()

        self.agent = None
        self.dummy_observation = "** Simplified html **"

        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        
        super().__init__(
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            # sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=formatted_prompt,
            few_shot=few_shot
        )
        self.eval_dataset = None
        self.max_steps = max_steps
        # self.rubric = WebArenaRubric()

        self.llm_parser = XMLParser(fields=["think", ("answer")])
        self.env_parser = XMLParser(fields=["result"])

        self.reward_funcs = [
            # self.llm_as_judge_reward_func,
            self.outcome_reward_func,
            # self.llm_parser.get_format_reward_func(),
        ]

    def init_env(self, **kwargs: Any) -> Any:

        args = config()
        args.sleep_after_execution = 3.0

        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True
        args.current_viewport_only = True

        # args.instruction_path = self.webarena_path + 'agent/prompts/jsons/p_webrl_chat_think.json'
        # args.result_dir = 'results_debug'
        # args.action_set_tag = 'webrl_id'
        # args.max_tokens = 512

        args.observation_type = 'webrl'
        args.max_obs_length = 0
        args.viewport_width = 1280
        args.viewport_height = 720
        args.temperature = 1.0
    
        # prepare(args)

        env = ScriptBrowserEnv( # TODO: this need to be updated to incorporate multiple context sessions (i.e., browser windows; n_windows = 1 by default)
            headless=not args.render,
            slow_mo=args.slow_mo,
            observation_type=args.observation_type,
            current_viewport_only=args.current_viewport_only,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=args.save_trace_enabled,
            sleep_after_execution=args.sleep_after_execution,
            captioning_fn=None,
            n_contexts=self.n_contexts,
        )

        return env

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n)) # type: ignore
        return self.eval_dataset
    
    # def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
    #     return self.rubric.get_reward_funcs()
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.get_reward_funcs()
    
    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs

    def get_last_answer(self, trajectory: List[Dict[str, str]]) -> str | None:
        """Extract the last answer from a trajectory."""
        for msg in reversed(trajectory):
            if msg['role'] == 'assistant':
                if self.llm_parser is None:
                    raise ValueError("Parser is not set")
                parsed = self.llm_parser.parse(msg['content'])
                if hasattr(parsed, 'answer') and parsed.answer is not None:
                    return parsed.answer
        return None
    
    def exact_answer_reward_func(self, prompts, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""

        # print(f'\n>>> completions: {completions}')
        # print(f'\n>>> answer: {answer}')

        # print(f'\n\n>>> How many completions: {len(completions)} asking for reward\n\n') # debug purpose
        # print(f'\n\n>>> How many prompts: {len(prompts)} asking for reward\n\n{json.dumps(prompts[0], indent=4)}') # debug purpose
        # print(f'\n\n>>> How many envs: {len(envs)} asking for reward\n\n') # debug purpose

        # print(f'>>>\n length of pages: {len(kwargs["pages"])}, type: {type(kwargs["pages"][0])}')

        responses = [self.get_last_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def outcome_reward_func(self, llm_as_judge_reward, **kwargs) -> List[float]:
        return llm_as_judge_reward

    def llm_as_judge_reward_func(self, completion: List[Dict[str, str]], prompt: dict, context_id: int, cur_obs: str, is_think_format=True, **kwargs) -> float:
        
        # print(f'\n\n>>> How many completions: {len(completion)} asking for reward\n\n') # debug purpose
        # print(f'\n\n>>> How many prompts: {len(prompt)} asking for reward\n\n') # debug purpose
        # print(f'\n\n>>> How many contexts in env: {len(self.env.contexts)} asking for reward\n\n') # debug purpose
        # print(f'\n\n>>> How many context_id: {context_id} asking for reward\n\n') # debug purpose

        config_obj = prompt

        eval_types = config_obj["eval"]["eval_types"]
        evaluator = evaluator_router(config_file="", eval_types=eval_types)

        if is_think_format:
            last_action = self.get_last_answer(completion)
            exit_message = None
            if last_action is not None:
                exit_message = re.search(r'exit\(message="(.*)"\)', last_action)
                # print(f'\n>>> last_action: {last_action}') # debug purpose
                # print(f'\n>>> exit_message: {exit_message}') # debug purpose
        else:
            assert completion[-1]["role"] == "assistant"
            last_action = completion[-1]["content"]
            exit_message = re.search(r'exit\(message="(.*)"\)', last_action)

        exit_message = exit_message.group(1) if exit_message is not None else "N/A"

        # print(f'\n>>> last_action: {last_action}') # debug purpose
        # print(f'\n>>> exit_message: {exit_message}') # debug purpose

        # print(f'\n>>> completions: {json.dumps(completion)} with exit_message: "{exit_message}"') # debug purpose
        # print(f'\n>>> context_id: {context_id}')

        # last_observation = self.env._get_obs(context_id)["text"]

        score = evaluator(
            trajectory=[],
            config_file="",
            exit_message=exit_message,
            config_obj=config_obj, 
            page=self.env._get_page(context_id)
        )  


        return score


    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
                # try:
                #     parsed = self.llm_parser.parse(message["content"], return_raw=True)
                #     # if hasattr(parsed, 'tool') and parsed.tool is not None:
                #     if hasattr(parsed, 'answer') and bool(re.match(r'^exit\(message=".*?"\)$', parsed.answer.strip())) is False:
                #         step_count += 1
                # except Exception:
                #     pass
        return step_count
    
    def is_completed(self, messages: List[Dict[str, str]], is_thinking_format=True, **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count >= self.max_steps:
                return True
            
            if is_thinking_format:
                parsed = self.llm_parser.parse(messages[-1]["content"])
                # Check if we got a valid answer field (not just None from failed parsing)
                # return hasattr(parsed, 'answer') and parsed.answer is not None
            
                return hasattr(parsed, 'answer') and bool(re.match(r'^exit\(message=".*?"\)$', parsed.answer.strip())) is True
            else:
                assert messages[-1]["role"] == "assistant"
                last_action = messages[-1]["content"]
                # print(f'\n>>> last_action in checking is_completed: {last_action}') # debug purpose
                return bool(re.search(r'exit\(message=".*?"\)$', last_action.strip())) is True
        except Exception:
            return False

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url
    
    def env_response(self, messages: List[Dict[str, str]], context_id, is_thinking_format=True, **kwargs: Any) -> Dict[str, str]: # add env.step() here
        # print(f'env step for context_id: {context_id} with step count {self._get_step_count(messages)}') # debug purpose

        try:
            response = messages[-1]["content"]            
            if is_thinking_format is True:
                parsed_response = self.llm_parser.parse(response)
                # print(f'\n>>> parsed_response: {parsed_response}') # debug purpose
                # print(f'\n>>> parsed.answer: {parsed_response.answer}') # debug purpose
                if hasattr(parsed_response, 'answer') and parsed_response.answer is not None:
                    parsed_action_str = self.map_url_to_local(parsed_response.answer)
                    action = create_webrl_id_based_action(parsed_action_str)
                    action["raw_prediction"] = parsed_response.answer
                    obs, _, _, _, info = self.env.step(action, context_id)

                    next_msg = {"role": "user", "content": ""}
                    if info["fail_error"] != "":
                        # print(f'\n>>> Web Error: {info["fail_error"]} after executing action {parsed_action_str}')
                        next_msg["content"] += f"Error: invalid action {parsed_action_str}.\nMake sure there is only one action enclosed by <answer> and </answer>."
                else:
                    next_msg = {"role": "user", "content": f"Error: cannot parse action from generated contents.\nMake sure there is only one action enclosed by <answer> and </answer>."}
                    # next_msg = {"role": "user", "content": ""}
                    obs = None
            else:
                parsed_action_str = self.map_url_to_local(response)
                action = create_webrl_id_based_action(parsed_action_str)
                action["raw_prediction"] = response
                obs, _, _, _, info = self.env.step(action, context_id)
                next_msg = {"role": "user", "content": ""}
                if info["fail_error"] != "":
                    next_msg["content"] += f"Error: invalid action {parsed_action_str}.\nFail error: {info['fail_error']}"
        except Exception as e:
            # print(f'Error: {str(e)}')
            next_msg = {"role": "user", "content": f"Error: invalid action {parsed_action_str}\nError: {e}"}
            # next_msg = {"role": "user", "content": ""}
            obs = None
        
        # if obs is not None:
        #     print(f'\n>>> Executing action: {action}') # debug purpose
        # else:
        #     print(f'\n>>> Executing action: Error - No action performed')

        return next_msg, obs

    def step_async(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams,
             tokenizer=None,
             task_configs=None,
             mask_thinking=True) -> List[Dict[str, Any],]:
        
        # if terminated, no need to repalce observation with ** Simplified html ** 
        # otherwise, need to replace observation with simplified html --> update completion ids and mask
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]

        print(f'\n======== live_indices: {live_indices} with traj length: {[self._get_step_count(states[i]["messages"]) for i in live_indices]} ===========') 

        def add_observation(messages, observation):
            round = self._get_step_count(messages)
            if round == 0:
                messages[-1]['content'] += f'\n\nRound {round}\n\n' + observation
            else:
                messages[-1]['content'] += f'Round {round}\n\n' + observation
            return messages
        
        messages_to_step = [add_observation(states[i]["messages"], states[i]["cur_observation"]) for i in live_indices]
        
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        # for i, j in enumerate(live_indices):
        def update_state(j, llm_response, task_config):
            state = states[j].copy()
            print(f'\n>>> processing live index: {j}') # debug purpose
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text}) # add model response to messages

            if self.is_completed(state["messages"]):
                '''if complted, no need to interact with env, no need to adjust completion ids and mask'''
                state["completed"] = True
                if len(state["prompt_ids"]) == 0: # if terminated at the first step
                    state["prompt_ids"] = llm_response.prompt_token_ids # prefix is system + task + init_observation
                
                prefix_len = len(state["prompt_ids"])
                total_prev_len = prefix_len + len(state["completion_ids"])

            else: # if not completed, adjust completion ids and mask with simplified html, and interact with env to get next observation
                state["messages"][-2]["content"] = state["messages"][-2]["content"].replace(state["cur_observation"], self.dummy_observation)

                if len(state["prompt_ids"]) == 0:
                    init_observation_ids = tokenizer.encode(state["init_observation"])
                    dummy_observation_ids = tokenizer.encode(self.dummy_observation)
                    state["prompt_ids"] = llm_response.prompt_token_ids[:-(len(init_observation_ids) + 5)] + dummy_observation_ids + llm_response.prompt_token_ids[-5:] 
                    prefix_len = len(llm_response.prompt_token_ids)
                    total_prev_len = prefix_len
                else:
                    prefix_len = len(state["prompt_ids"])
                    total_prev_len = prefix_len + len(state["completion_ids"])
        
            # get token lengths of env response and new completion
            new_completion_len = len(llm_response.outputs[0].token_ids)
            new_copletion_think_len = len(tokenizer.encode(llm_response.outputs[0].text.split('<answer>')[0])) if mask_thinking else 0
            new_completion_answer_len = new_completion_len - new_copletion_think_len

            # update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids) # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][prefix_len:]

            if state["completed"] is True:
                env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len
                state["reward"] = self.llm_as_judge_reward_func(completion=state["messages"], prompt=task_config, context_id=j, cur_obs=state["cur_observation"]) # TODO: this is incorrect, each prompt should have its own env instance, rather than sharing the same env instance
                state["n_steps"] = self._get_step_count(state["messages"])
            else:
                # this need to be adjusted with dummy observation
                env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len
                
                # replace old observation ids with dummy observation ids
                if prefix_len != len(list(llm_response.prompt_token_ids)):
                    cur_observation_ids = tokenizer.encode(state["cur_observation"])
                    dummy_observation_ids = tokenizer.encode(self.dummy_observation)

                    # 5 connecting tokens ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
                    state["completion_ids"] = state["completion_ids"][:-(new_completion_len + len(cur_observation_ids) + 5)] + dummy_observation_ids + state["completion_ids"][-(new_completion_len + 5):] 

                    env_response_len -= len(cur_observation_ids) - len(dummy_observation_ids)

                # parse action and interact with env to get new observation
                # print(f'\n>>> observation before env_response: {state["cur_observation"]}') # debug purpose

                next_msg, obs = self.env_response(state["messages"], context_id=j)
                state["messages"].append(next_msg)
                if obs is not None:
                    state["cur_observation"] = obs["text"]
                
                # print(f'\n>>> observation after env_response: {state["cur_observation"]}') # debug purpose

            # update completion masks
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([self.env_mask] * new_copletion_think_len)
            state["completion_mask"].extend([1] * new_completion_answer_len)

            # print(f'current completion mask length: {len(state["completion_mask"])}, sum: {sum(state["completion_mask"])}') # debug purpose

            assert len(state["completion_mask"]) == len(state["completion_ids"])

            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i], task_configs[i]) for i, j in enumerate(live_indices)]
            ))

        for j, state in results:
            states[j] = state

        return states

    def step_webrl_async(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams,
             tokenizer=None,
             task_configs=None) -> List[Dict[str, Any],]:
        
        # if terminated, no need to repalce observation with ** Simplified html ** 
        # otherwise, need to replace observation with simplified html --> update completion ids and mask
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]

        print(f'\n======== live_indices: {live_indices} with traj length: {[self._get_step_count(states[i]["messages"]) for i in live_indices]} ===========') 
        
        def add_observation(messages, observation):
            round = self._get_step_count(messages)
            if round == 0:
                messages[-1]['content'] += f'\n\nRound {round}\n\n' + observation
            else:
                messages[-1]['content'] += f'Round {round}\n\n' + observation
            return messages
        
        messages_to_step = [add_observation(states[i]["messages"], states[i]["cur_observation"]) for i in live_indices]
        
        messages_to_step = [apply_webrl_format(msg) for msg in messages_to_step]

        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        dummy_observation_ids = tokenizer.encode(self.dummy_observation)

        # for i, j in enumerate(live_indices):
        def update_state(j, llm_response, task_config):
            state = states[j].copy()
            print(f'\n>>> processing live index: {j}') # debug purpose
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text}) # add model response to messages

            if self.is_completed(state["messages"], is_thinking_format=False):
                '''if complted, no need to interact with env, no need to adjust completion ids and mask'''
                state["completed"] = True
                if len(state["prompt_ids"]) == 0: # if terminated at the first step
                    state["prompt_ids"] = llm_response.prompt_token_ids # prefix is system + task + init_observation
                
                prefix_len = len(state["prompt_ids"])
                total_prev_len = prefix_len + len(state["completion_ids"])

            else: # if not completed, adjust completion ids and mask with simplified html, and interact with env to get next observation
                state["messages"][-2]["content"] = state["messages"][-2]["content"].replace(state["cur_observation"], self.dummy_observation)

                if len(state["prompt_ids"]) == 0:
                    init_observation_ids = tokenizer.encode(state["init_observation"])
                    # dummy_observation_ids = tokenizer.encode(self.dummy_observation)
                    state["prompt_ids"] = llm_response.prompt_token_ids[:-(len(init_observation_ids) + 5)] + dummy_observation_ids + llm_response.prompt_token_ids[-5:] 
                    prefix_len = len(llm_response.prompt_token_ids)
                    total_prev_len = prefix_len
                else:
                    if len(state["messages"]) == 4:
                        intent_ids = state["prompt_ids"][4:-(len(dummy_observation_ids) + 14)]
                        state["prompt_ids"] = state["prompt_ids"][:-(len(dummy_observation_ids) + 5)] + intent_ids + state["prompt_ids"][-5:]

                    prefix_len = len(state["prompt_ids"])
                    total_prev_len = prefix_len + len(state["completion_ids"])
        
            # get token lengths of env response and new completion
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids) # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][prefix_len:]

            if state["completed"] is True:
                env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len
                state["reward"] = self.llm_as_judge_reward_func(completion=state["messages"], prompt=task_config, context_id=j, cur_obs=state["cur_observation"], is_think_format=False) # TODO: this is incorrect, each prompt should have its own env instance, rather than sharing the same env instance
                state["n_steps"] = self._get_step_count(state["messages"])
            else:
                # this need to be adjusted with dummy observation
                env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len
                
                # replace old observation ids with dummy observation ids
                if prefix_len != len(list(llm_response.prompt_token_ids)):
                    cur_observation_ids = tokenizer.encode(state["cur_observation"])
                    # dummy_observation_ids = tokenizer.encode(self.dummy_observation)

                    # 5 connecting tokens ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
                    state["completion_ids"] = state["completion_ids"][:-(new_completion_len + len(cur_observation_ids) + 5)] + dummy_observation_ids + state["completion_ids"][-(new_completion_len + 5):] 

                    env_response_len -= len(cur_observation_ids) - len(dummy_observation_ids)

                # parse action and interact with env to get new observation
                # print(f'\n>>> observation before env_response: {state["cur_observation"]}') # debug purpose

                next_msg, obs = self.env_response(state["messages"], context_id=j, is_thinking_format=False)
                state["messages"].append(next_msg)
                if obs is not None:
                    state["cur_observation"] = obs["text"]
                
                # print(f'\n>>> observation after env_response: {state["cur_observation"]}') # debug purpose

            # update completion masks
            if len(state["completion_mask"]) != 0 and state["completed"] is False:
                env_response_len += 1
                new_completion_len -= 1

            state["completion_mask"].extend([self.env_mask] * (env_response_len))

            if len(state["completion_mask"]) != 0:
                state["completion_mask"].extend([1] * (new_completion_len))
            else:
                state["completion_mask"] = [1] * (new_completion_len - 1)
                state["completion_mask"].extend([self.env_mask] * 1)

            assert len(state["completion_mask"]) == len(state["completion_ids"])

            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i], task_configs[i]) for i, j in enumerate(live_indices)]
            ))

        for j, state in results:
            states[j] = state

        return states


    def step_webrl(self,
            states: List[Dict[str, Any]],
            llm: LLM,
            sampling_params: SamplingParams,
            tokenizer=None,
            task_configs=None) -> List[Dict[str, Any],]:
        
        # if terminated, no need to repalce observation with ** Simplified html ** 
        # otherwise, need to replace observation with simplified html --> update completion ids and mask
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]

        print(f'\n======== live_indices: {live_indices} with traj length: {[self._get_step_count(states[i]["messages"]) for i in live_indices]} ===========') 

        def add_observation(messages, observation):
            round = self._get_step_count(messages)
            if round == 0:
                messages[-1]['content'] += f'\n\nRound {round}\n\n' + observation
            else:
                messages[-1]['content'] += f'Round {round}\n\n' + observation
            return messages
        
        messages_to_step = [add_observation(states[i]["messages"], states[i]["cur_observation"]) for i in live_indices]

        # print(f'\n>>> messages_to_step: {json.dumps(messages_to_step, indent=4)}') # debug purpose

        messages_to_step = [apply_webrl_format(msg) for msg in messages_to_step]

        # print(f'\n>>> messages_to_step (webrl format): {json.dumps(messages_to_step, indent=4)}') # debug purpose
        

        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        dummy_observation_ids = tokenizer.encode(self.dummy_observation)

        for i, j in enumerate(live_indices):
            print(f'\n>>> processing live index: {j}') # debug purpose
            states[j]["messages"].append({"role": "assistant", "content": llm_responses[i].outputs[0].text}) # add model response to messages

            # print(f'>>> Current prompt ids (First): {json.dumps(tokenizer.decode(states[j]["prompt_ids"]), indent=4)}') # debug purpose

            if self.is_completed(states[j]["messages"], is_thinking_format=False):
                '''if complted, no need to interact with env, no need to adjust completion ids and mask'''
                states[j]["completed"] = True
                if len(states[j]["prompt_ids"]) == 0: # if terminated at the first step
                    states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids # prefix is system + task + init_observation
                
                prefix_len = len(states[j]["prompt_ids"])
                total_prev_len = prefix_len + len(states[j]["completion_ids"])

            else: # if not completed, adjust completion ids and mask with simplified html, and interact with env to get next observation
                states[j]["messages"][-2]["content"] = states[j]["messages"][-2]["content"].replace(states[j]["cur_observation"], self.dummy_observation)

                if len(states[j]["prompt_ids"]) == 0:
                    init_observation_ids = tokenizer.encode(states[j]["init_observation"])
                    # dummy_observation_ids = tokenizer.encode(self.dummy_observation)
                    states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids[:-(len(init_observation_ids) + 5)] + dummy_observation_ids + llm_responses[i].prompt_token_ids[-5:] 
                    prefix_len = len(llm_responses[i].prompt_token_ids)
                    # print(f'>>> Prefix: {json.dumps(tokenizer.decode(llm_responses[i].prompt_token_ids))}') # debug purpose
                    total_prev_len = prefix_len
                else:
                    if len(states[j]["messages"]) == 4:
                        # 4 for ['<|begin_of_text|>', 'Task', 'Instruction', ':']
                        intent_ids = states[j]["prompt_ids"][4:-(len(dummy_observation_ids) + 14)]
                        # print(f'>>> Intent ids: {intent_ids}\ntokens:{tokenizer.convert_ids_to_tokens(intent_ids)}\n{json.dumps(tokenizer.decode(intent_ids))}')
                        # intent_ids = states[j]["prompt_ids"][3:-(len(dummy_observation_ids) + 14)]
                        states[j]["prompt_ids"] = states[j]["prompt_ids"][:-(len(dummy_observation_ids) + 5)] + intent_ids + states[j]["prompt_ids"][-5:]
                        
                    prefix_len = len(states[j]["prompt_ids"])
                    total_prev_len = prefix_len + len(states[j]["completion_ids"])

            # print(f'>>> Current prompt ids (Second): {json.dumps(tokenizer.decode(states[j]["prompt_ids"]), indent=4)}') # debug purpose

            # get token lengths of env response and new completion
            new_completion_len = len(llm_responses[i].outputs[0].token_ids)
            # print(f'>>> Current new completion length: {new_completion_len}:\n{llm_responses[i].outputs[0].token_ids}\n{tokenizer.convert_ids_to_tokens(llm_responses[i].outputs[0].token_ids)}\n{json.dumps(tokenizer.decode(llm_responses[i].outputs[0].token_ids), indent=4)}') # debug purpose

            # update completion ids
            states[j]["completion_ids"] = list(llm_responses[i].prompt_token_ids) # type: ignore
            states[j]["completion_ids"].extend(list(llm_responses[i].outputs[0].token_ids))

            # print(f'>>> Current completion ids (First): {json.dumps(tokenizer.decode(states[j]["completion_ids"]), indent=4)}') # debug purpose

            states[j]["completion_ids"] = states[j]["completion_ids"][prefix_len:]

            # print(f'>>> Current completion ids (Second): {json.dumps(tokenizer.decode(states[j]["completion_ids"]), indent=4)}') # debug purpose

            # states[j]["completion_ids"] might contains old observation ids, need to replace it with dummy observation ids when states[j]["completed"] is False
            if states[j]["completed"] is True:
                env_response_len  = len(list(llm_responses[i].prompt_token_ids)) - total_prev_len
                states[j]["reward"] = self.llm_as_judge_reward_func(completion=states[j]["messages"], prompt=task_configs[j], context_id=j, cur_obs=states[j]["cur_observation"], is_think_format=False) # TODO: this is incorrect, each prompt should have its own env instance, rather than sharing the same env instance
                states[j]["n_steps"] = self._get_step_count(states[j]["messages"])
            else:
                # this need to be adjusted with dummy observation
                env_response_len  = len(list(llm_responses[i].prompt_token_ids)) - total_prev_len
                
                # replace old observation ids with dummy observation ids
                if prefix_len != len(list(llm_responses[i].prompt_token_ids)):
                    cur_observation_ids = tokenizer.encode(states[j]["cur_observation"])
                    # dummy_observation_ids = tokenizer.encode(self.dummy_observation)

                    # 5 connecting tokens ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
                    states[j]["completion_ids"] = states[j]["completion_ids"][:-(new_completion_len + len(cur_observation_ids) + 5)] + dummy_observation_ids + states[j]["completion_ids"][-(new_completion_len + 5):] 

                    env_response_len -= len(cur_observation_ids) - len(dummy_observation_ids)

                # parse action and interact with env to get new observation
                # print(f'\n>>> observation before env_response: {states[j]["cur_observation"]}') # debug purpose

                next_msg, obs = self.env_response(states[j]["messages"], context_id=j, is_thinking_format=False)
                states[j]["messages"].append(next_msg)
                if obs is not None:
                    states[j]["cur_observation"] = obs["text"]
                
                # print(f'\n>>> observation after env_response: {states[j]["cur_observation"]}') # debug purpose

            # print(f'>>> Current completion ids (Third): {json.dumps(tokenizer.decode(states[j]["completion_ids"]), indent=4)}') # debug purpose

            # update completion masks
            if len(states[j]["completion_mask"]) != 0 and states[j]["completed"] is False:
                env_response_len += 1
                new_completion_len -= 1

            states[j]["completion_mask"].extend([self.env_mask] * (env_response_len))

            if len(states[j]["completion_mask"]) != 0:
                states[j]["completion_mask"].extend([1] * (new_completion_len))
            else:
                states[j]["completion_mask"] = [1] * (new_completion_len - 1)
                states[j]["completion_mask"].extend([self.env_mask] * 1)

            # print(f'current completion mask length: {len(states[j]["completion_mask"])}, sum: {sum(states[j]["completion_mask"])}') # debug purpose

            assert len(states[j]["completion_mask"]) == len(states[j]["completion_ids"])

        return states
    
    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams,
             tokenizer=None,
             task_configs=None,
             mask_thinking=True) -> List[Dict[str, Any],]:
        
        # if terminated, no need to repalce observation with ** Simplified html ** 
        # otherwise, need to replace observation with simplified html --> update completion ids and mask
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]

        print(f'\n======== live_indices: {live_indices} with traj length: {[self._get_step_count(states[i]["messages"]) for i in live_indices]} ===========') 

        def add_observation(messages, observation):
            round = self._get_step_count(messages)
            if round == 0:
                messages[-1]['content'] += f'\n\nRound {round}\n\n' + observation
            else:
                messages[-1]['content'] += f'Round {round}\n\n' + observation
            return messages
        
        messages_to_step = [add_observation(states[i]["messages"], states[i]["cur_observation"]) for i in live_indices]

        # print(f'\n>>> messages_to_step: {json.dumps(messages_to_step, indent=4)}') # debug purpose

        # print(f'\n>>> messages_to_step (webrl format): {json.dumps(messages_to_step, indent=4)}') # debug purpose
        
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        for i, j in enumerate(live_indices):
            print(f'\n>>> processing live index: {j}') # debug purpose
            states[j]["messages"].append({"role": "assistant", "content": llm_responses[i].outputs[0].text}) # add model response to messages

            if self.is_completed(states[j]["messages"]):
                '''if complted, no need to interact with env, no need to adjust completion ids and mask'''
                states[j]["completed"] = True
                if len(states[j]["prompt_ids"]) == 0: # if terminated at the first step
                    states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids # prefix is system + task + init_observation
                
                prefix_len = len(states[j]["prompt_ids"])
                total_prev_len = prefix_len + len(states[j]["completion_ids"])

            else: # if not completed, adjust completion ids and mask with simplified html, and interact with env to get next observation
                states[j]["messages"][-2]["content"] = states[j]["messages"][-2]["content"].replace(states[j]["cur_observation"], self.dummy_observation)

                if len(states[j]["prompt_ids"]) == 0:
                    init_observation_ids = tokenizer.encode(states[j]["init_observation"])
                    dummy_observation_ids = tokenizer.encode(self.dummy_observation)
                    states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids[:-(len(init_observation_ids) + 5)] + dummy_observation_ids + llm_responses[i].prompt_token_ids[-5:] 
                    prefix_len = len(llm_responses[i].prompt_token_ids)
                    total_prev_len = prefix_len
                else:
                    prefix_len = len(states[j]["prompt_ids"])
                    total_prev_len = prefix_len + len(states[j]["completion_ids"])
        
            # get token lengths of env response and new completion
            new_completion_len = len(llm_responses[i].outputs[0].token_ids)
            new_copletion_think_len = len(tokenizer.encode(llm_responses[i].outputs[0].text.split('<answer>')[0])) if mask_thinking else 0
            new_completion_answer_len = new_completion_len - new_copletion_think_len

            # update completion ids
            states[j]["completion_ids"] = list(llm_responses[i].prompt_token_ids) # type: ignore
            states[j]["completion_ids"].extend(list(llm_responses[i].outputs[0].token_ids))
            states[j]["completion_ids"] = states[j]["completion_ids"][prefix_len:]

            # states[j]["completion_ids"] might contains old observation ids, need to replace it with dummy observation ids when states[j]["completed"] is False
            if states[j]["completed"] is True:
                env_response_len  = len(list(llm_responses[i].prompt_token_ids)) - total_prev_len
                states[j]["reward"] = self.llm_as_judge_reward_func(completion=states[j]["messages"], prompt=task_configs[j], context_id=j, cur_obs=states[j]["cur_observation"]) # TODO: this is incorrect, each prompt should have its own env instance, rather than sharing the same env instance
                states[j]["n_steps"] = self._get_step_count(states[j]["messages"])
            else:
                # this need to be adjusted with dummy observation
                env_response_len  = len(list(llm_responses[i].prompt_token_ids)) - total_prev_len
                
                # replace old observation ids with dummy observation ids
                if prefix_len != len(list(llm_responses[i].prompt_token_ids)):
                    cur_observation_ids = tokenizer.encode(states[j]["cur_observation"])
                    dummy_observation_ids = tokenizer.encode(self.dummy_observation)

                    # 5 connecting tokens ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
                    states[j]["completion_ids"] = states[j]["completion_ids"][:-(new_completion_len + len(cur_observation_ids) + 5)] + dummy_observation_ids + states[j]["completion_ids"][-(new_completion_len + 5):] 

                    env_response_len -= len(cur_observation_ids) - len(dummy_observation_ids)

                # parse action and interact with env to get new observation
                # print(f'\n>>> observation before env_response: {states[j]["cur_observation"]}') # debug purpose

                next_msg, obs = self.env_response(states[j]["messages"], context_id=j)
                states[j]["messages"].append(next_msg)
                if obs is not None:
                    states[j]["cur_observation"] = obs["text"]
                
                # print(f'\n>>> observation after env_response: {states[j]["cur_observation"]}') # debug purpose

            # update completion masks
            states[j]["completion_mask"].extend([self.env_mask] * env_response_len)
            states[j]["completion_mask"].extend([self.env_mask] * new_copletion_think_len)
            states[j]["completion_mask"].extend([1] * new_completion_answer_len)

            # print(f'current completion mask length: {len(states[j]["completion_mask"])}, sum: {sum(states[j]["completion_mask"])}') # debug purpose

            assert len(states[j]["completion_mask"]) == len(states[j]["completion_ids"])

        return states

    def get_init_observation(self, tasks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        states = []
        # print(f'>>> len(tasks): {len(tasks)}') # debug purpose
        # print(f'>>> # of context_window: {self.env.n_contexts}') # debug purpose
        
        is_reset = False
        for idx, task in enumerate(tasks):
            state = {
                "messages": task['prompt'],
                "cur_observation": "",
                "init_observation": "",
                "storage_state": task['storage_state'],
                "prompt_messages": len(task['prompt']),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
                "reward": 0.0,
                "n_steps": 0
            }

            if state["storage_state"]:
                cookie_file_name = os.path.basename(state["storage_state"])
                comb = get_site_comb_from_filepath(cookie_file_name)
                temp_dir = tempfile.mkdtemp()

                subprocess.run(
                    [
                        "python",
                        # "auto_login.py",
                        "/workspace/verifiers/verifiers/envs/WebArena/auto_login.py",
                        "--auth_folder",
                        temp_dir,
                        "--site_list",
                        *comb,
                    ]
                )
                state["storage_state"] = f"{temp_dir}/{cookie_file_name}"

            task["storage_state"] = state["storage_state"]
            if not is_reset:
                print(f'\n\n=====reset all context_windows ({len(tasks)} windows)=====') # debug purpose
                obs, info = self.env.reset(options={"config_obj": task})
                is_reset = True
            state["init_observation"] = obs["text"]
            state["cur_observation"] = state["init_observation"]

            states.append(state)

        return states
    
    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 task_configs=None,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        
        tokenizer = llm.get_tokenizer()

        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        # states = [{
        #     "messages": m['prompt'],
        #     "observation": "",
        #     "storage_state": m['storage_state'],
        #     "prompt_messages": len(m['prompt']),
        #     "prompt_ids": [],
        #     "completed": False,
        #     "completion_ids": [],
        #     "completion_mask": []
        # } for m in prompts]

        # print(f'\n\n====type of task_configs===: {type(task_configs)}') # debug purpose
        # print(f'task_configs: {task_configs}') # debug purpose

        # get_init_observation
        states = self.get_init_observation(task_configs)

        # main loop
        idx = 0
        while not all_completed:

            states = self.step_webrl(states, llm, custom_sp, tokenizer, task_configs)  

            all_completed = all(state["completed"] for state in states)
            idx += 1
            
        # time.sleep(3) # 

        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        completion_reward = [s["reward"] for s in states]
        completion_steps = [s["n_steps"] for s in states]

        # print(f'\n\n>>>>>> shape of completion_ids: ({len(completion_ids)}, {len(completion_ids[0])})')

        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "reward": completion_reward,
            "steps": completion_steps
        }
        
        return output