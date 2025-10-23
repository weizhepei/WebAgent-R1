"""Calculate the breakdown of success rate by different brekadown"""

import json
import os
import argparse

TOTAL_N_TASKS=165 # 812

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config file",
        default="config_files/wa/test_webarena_lite.json",
    )

    args = parser.parse_args()
    return args


def calc_sr(id_to_success: dict[int, bool]) -> dict[str, float]:
    sr = {
        "total": len(id_to_success),
        "success": sum(id_to_success.values()),
        "fail": len(id_to_success) - sum(id_to_success.values()),
    }

    # keep 2 decimal places
    sr["success_rate"] = sr["success_rate"] = round(sr["success"] / sr["total"], 4) * 100 if sr["total"] != 0 else None 


    return sr


def parse_result(log_file: str) -> dict[int, bool]:
    id_to_success: dict[int, bool] = {}

    with open(log_file, "r") as f:
        for sub_file in f:
            print(sub_file)
            sub_file = sub_file.strip()
            if len(sub_file) > 0:
                # parse
                with open(sub_file, "r") as f_2:
                    for line in f_2:
                        if "[Config file]" in line:
                            id = os.path.basename(line.split()[-1]).split(".")[0]
                        elif "[Result]" in line:
                            if "(FAIL)" in line:
                                success = False
                            elif "(PASS)" in line:
                                success = True
                            else:
                                raise ValueError(f"Unknown result: {line}")
                            id_to_success[int(id)] = success
    
    return id_to_success


def main(args: argparse.Namespace):
    id_to_success = parse_result(args.log_file)

    print(f'missing task_ids: {[i for i in range(TOTAL_N_TASKS) if i not in id_to_success]}')

    overall_sr = calc_sr(id_to_success)

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # get achievable and unachievable tasks
    non_achievable = set()
    for task in config:
        task_id = int(task["task_id"])
        if (
            task["eval"]["eval_types"] == ["string_match"]
            and task["eval"]["reference_answers"].get("fuzzy_match", "") == "N/A"
        ):
            non_achievable.add(task_id)

    achievable_sr = calc_sr(
        {k: v for k, v in id_to_success.items() if k not in non_achievable}
    )
    unachievable_sr = calc_sr(
        {k: v for k, v in id_to_success.items() if k in non_achievable}
    )

    # get sr per website
    website_to_ids = {
        website: set()
        for website in [
            "shopping",
            "shopping_admin",
            "gitlab",
            "reddit",
            "map",
            "wikipedia",
        ]
    }
    
    for task in config:
        task_id = int(task["task_id"])
        cur_sites = task["sites"]
        for site in cur_sites:
            website_to_ids[site].add(task_id)

    print(f'website_to_ids: {website_to_ids}')

    website_sr = {
        website: calc_sr(
            {k: v for k, v in id_to_success.items() if k in website_to_ids[website]}
        )
        for website in website_to_ids
    }

    # by task type
    task_type_to_ids = {
        "info_seeking": set(),
        "site_nav": set(),
        "content_config": set(),
    }
    for task in config:
        task_id = int(task["task_id"])
        if task["eval"]["eval_types"] == ["string_match"]:
            task_type_to_ids["info_seeking"].add(task_id)
        elif task["eval"]["eval_types"] == ["url_match"]:
            task_type_to_ids["site_nav"].add(task_id)
        else:
            task_type_to_ids["content_config"].add(task_id)

    task_type_sr = {
        task_type: calc_sr(
            {k: v for k, v in id_to_success.items() if k in task_type_to_ids[task_type]}
        )
        for task_type in task_type_to_ids
    }

    print("=====================================")
    print("Overall SR:")
    print(overall_sr)
    print("=====================================")
    print("Achievable SR:")
    print(achievable_sr)
    print("=====================================")
    print("Unachievable SR:")
    print(unachievable_sr)
    print("=====================================")
    print("Website SR:")
    for website, sr in website_sr.items():
        print(website)
        print(sr)
    print("=====================================")
    print("Task type SR:")
    for task_type, sr in task_type_sr.items():
        print(task_type)
        print(sr)


# python scripts/calc_breakdown_sr.py --log_file /home/zhepei/VisualAgentBench/VAB-WebArena-Lite/eval_results/Qwen2.5-3B-Instruct/GRPO/conditional_reward_results_temp_1.0_reason_mode/log_files.txt

if __name__ == "__main__":
    args = config()
    main(args)
