import json
import os
import argparse
import numpy as np
from transformers import AutoTokenizer

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="Path to the log file")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config file",
        default="config_files/wa/test_webarena_lite.json",
        # default="config_files/wa/train_webarena_lite.json",
    )

    args = parser.parse_args()
    return args

def main(results_dir, tokenizer):
    # list all json files in the directory
    json_files = []
    for file in os.listdir(results_dir + '/actions'):
        if file.endswith(".json"):
            json_files.append(file)
    
    # sort the json files
    json_files.sort()

    print(f'How many json files: {len(json_files)}')

    rounds = []
    response_lengths = []

    # read each json file
    for file in json_files:
        with open(os.path.join(results_dir + '/actions', file), 'r') as f:
            data = json.load(f)
            # print(data)
            rounds.append(len(data['actions']))
            lengths = []
            for action in data['actions']:
                tokenized_action = tokenizer(action, return_tensors='pt')
                tokenized_len = len(tokenized_action['input_ids'][0])
                lengths.append(tokenized_len)
                # print(f'Action: {action}\n\nTokenized action: {tokenized_action}\n\nTokenized length: {tokenized_len}\n\n')
            
            if len(lengths) > 0:
                response_lengths.append(np.mean(lengths))
            else:
                response_lengths.append(0)
    
    # print the rounds and response lengths
    print(f'Average rounds for {len(rounds)} tasks: {np.mean(rounds)}')
    print(f'Average response lengths per round: {np.mean(response_lengths)}')


# python scripts/check_stats.py --log_dir <path_to_log_dir>


if __name__ == "__main__":
    args = config()

    # get tokenizer for Qwen/Qwen2.5-3B-Instruct
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

    main(args.log_dir, tokenizer)