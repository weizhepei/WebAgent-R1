import json
import copy

new_data = []
used_data = []

with open('test_webarena_lite.raw.json', 'r') as f:
    data = json.load(f)
    for d in data:
        if d['old_task_id'] not in used_data:
            used_data.append(d['old_task_id'])

print(f'how many used data: {len(used_data)}')

with open('test_webarena.raw.json', 'r') as f:
    data = json.load(f)
    for d in data:
        if d['task_id'] not in used_data:
            new_d = copy.deepcopy(d)
            new_d['old_task_id'] = new_d['task_id']
            new_d['task_id'] = len(new_data)
            new_data.append(new_d)

print(f'how many new data: {len(new_data)}')

print(f'how many total data: {len(new_data) + len(used_data)}; original data: {len(data)}')

with open('train_webarena_lite.raw.json', 'w') as f:
    json.dump(new_data, f, indent=4)

    
