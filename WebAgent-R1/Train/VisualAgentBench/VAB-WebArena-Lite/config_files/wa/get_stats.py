import json

with open(f'test_webarena_lite.raw.json', 'r') as f:
    data = f.read()
    for line in data:
        