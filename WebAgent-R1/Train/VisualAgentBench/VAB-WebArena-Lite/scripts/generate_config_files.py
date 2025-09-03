"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os

from browser_env.env_config import *


def main() -> None:
    # DATASET = os.environ["DATASET"]
    DATASET = 'webarena_rl_round_1'
    if DATASET == "webarena":
        print("DATASET: webarena")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"WIKIPEDIA: {WIKIPEDIA}")
        print(f"MAP: {MAP}")
        
        inp_paths = ["config_files/wa/test_webarena.raw.json", "config_files/wa/test_webarena_lite.raw.json"]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
        }
    elif DATASET == "webarena_rl_round_1":
        print("DATASET: webarena_rl_round_1")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"WIKIPEDIA: {WIKIPEDIA}")
        print(f"MAP: {MAP}")
        
        inp_paths = ["config_files/wa/sft_train_webarena.raw.json"]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
        }
    elif DATASET == "visualwebarena":
        print("DATASET: visualwebarena")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        inp_paths = [
            "config_files/vwa/test_classifieds.raw.json", "config_files/vwa/test_shopping.raw.json", "config_files/vwa/test_reddit.raw.json",
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
        }
    else:
        raise ValueError(f"Dataset not implemented: {DATASET}")
        
    for inp_path in inp_paths:
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        with open(inp_path, "r") as f:
            raw = f.read()
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
        data = json.loads(raw)
        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                template = {
                    "require_login": True,
                    "geolocation": None,
                    "intent_template": "",
                    "instantiation_dict": {},
                    "require_reset": None,
                    "eval": {
                        "eval_types": [
                        "string_match"
                        ],
                        "reference_answers": {
                        "exact_match": "N/A"
                        },
                        "reference_url": "",
                        "program_html": [],
                        "string_note": "",
                        "reference_answer_raw_annotation": ""
                    },
                    "intent_template_id": 0
                    }
                
                    # "intent": <Task>,
                    # "storage_state": "./.auth/shopping_admin_state.json",
                    #         "sites": [
                    #     <site> # possible choices: "shopping_admin", "map", "shopping", "reddit", "gitlab"
                    # ],
                    # "task_id": <Your task id>
                    # "start_url": <start url of site>, # possible choices: "__SHOPPING_ADMIN__", "__SHOPPING__", "__GITLAB__", "__MAP__", "__REDDIT__"
                
                if 'task_id' not in item:
                    item['task_id'] = idx
                if len(item['sites']) > 1:
                    item['storage_state'] = None
                elif item['sites'][0] == 'map':
                    item['storage_state'] = None
                else:
                    item['storage_state'] = f"./.auth/{item['sites'][0]}_state.json"

                template.update(item)
                json.dump(template, f, indent=2)


if __name__ == "__main__":
    main()
