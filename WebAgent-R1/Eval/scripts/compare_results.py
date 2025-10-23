import json
import os
import argparse
import numpy as np
from transformers import AutoTokenizer

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir1", type=str, help="Path to the log file")
    parser.add_argument("--res_dir2", type=str, help="Path to the log file")

    args = parser.parse_args()
    return args

def main(results_dir1, results_dir2):
    with open(results_dir1 + '/export.txt', 'r') as f1:
        lines_res_1 = f1.readlines()
    
    with open(results_dir2 + '/export.txt', 'r') as f2:
        lines_res_2 = f2.readlines()
    
    assert len(lines_res_1) == len(lines_res_2)

    mis_matched = []
    r1_success = set()
    r2_success = set()
    for idx, (r1,r2) in enumerate(zip(lines_res_1, lines_res_2)):
        if r1 != r2:
            mis_matched.append(idx)
            # print(f'>>>Task {idx} - r1: {type(r1)}{r1}=; r2: {type(r2)}{r2}=')
        if r1.strip() == '1.0':
            r1_success.add(idx)
        if r2.strip() == '1.0':
            r2_success.add(idx)

    
    print(f'Mismacthed tasks: {mis_matched} ({len(mis_matched)} out of {len(lines_res_1)} tasks)')

    print(f'How many success in res_1: {len(r1_success)} ({len(r1_success)/len(lines_res_1)}) ; unique success: {len(r1_success - r2_success)}: {sorted(list(r1_success - r2_success))}')
    print(f'How many success in res_2: {len(r2_success)} ({len(r2_success)/len(lines_res_2)}); unique success: {len(r2_success - r1_success)}: {sorted(list(r2_success - r1_success))}')

# python scripts/compare_results.py --res_dir1 <path_to_eval_result1> --res_dir2 <path_to_eval_result2>

if __name__ == "__main__":
    args = config()

    main(args.res_dir1, args.res_dir2)
