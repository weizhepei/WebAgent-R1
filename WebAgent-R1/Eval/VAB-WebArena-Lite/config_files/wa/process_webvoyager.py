import json


website2task = {}

with open('web_voyager_data.jsonl', 'r') as f:
    data = f.readlines()
    for d in data:
        d = json.loads(d)
        if d['web_name'] not in website2task:
            website2task[d['web_name']] = []
        website2task[d['web_name']].append(d)

print(f'how many websites: {len(website2task)}')

# how many tasks in each website
for k, v in website2task.items():
    print(f'{k}: {len(v)}')

# total number of tasks
total_tasks = 0
for k, v in website2task.items():
    total_tasks += len(v)
print(f'total tasks: {total_tasks}')

# select 3 tasks from each website and save to a new jsonl file
new_data = []
for k, v in website2task.items():
    if len(v) > 3:
        new_data.extend(v[:3])
    else:
        new_data.extend(v)
print(f'new data length: {len(new_data)}')

formatted_new_data = []
for d in new_data:
    new_d = {}
    new_d['intent'] = d['ques']
    new_d['sites'] = [d['web_name']]
    new_d['start_url'] = d['web']
    new_d['task_id'] = d['id']
    formatted_new_data.append(new_d)

print(f'formatted new data length: {len(formatted_new_data)}')

with open('test_webvoyager_lite.raw.json', 'w') as f:
    json.dump(formatted_new_data, f, indent=4)

# select all tasks from each website and save to a new jsonl file
new_data_all = []
for k, v in website2task.items():
    new_data_all.extend(v)
print(f'new data all length: {len(new_data_all)}')

formatted_new_data_all = []
for d in new_data_all:
    new_d = {}
    new_d['intent'] = d['ques']
    new_d['sites'] = [d['web_name']]
    new_d['start_url'] = d['web']
    new_d['task_id'] = d['id']
    formatted_new_data_all.append(new_d)

print(f'formatted new data all length: {len(formatted_new_data_all)}')

with open('test_webvoyager.raw.json', 'w') as f:
    json.dump(formatted_new_data_all, f, indent=4)




