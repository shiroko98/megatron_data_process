import json


with open("/mnt/data/Codes/json2binidx_tool/megatron/20241028192429/data_part_15.jsonl", "r") as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error in line {i+1}: {e}")
            break