import os
import csv
import json

def read_json(src):
    with open(src, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(obj, dst, indent=None):
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def read_jsonl(src):
    data = []
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(obj, dst):
    with open(dst, 'w', encoding='utf-8') as f:
        for item in obj:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_txt(src):
    with open(src, 'r', encoding='utf-8') as f:
        return f.read()

def write_txt(obj, dst):
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(obj)

def read_csv(src, delimiter='\t'):
    data = []
    with open(src, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(row)
    return data

def write_csv(obj, dst, delimiter='\t'):
    if not obj:
        return
    with open(dst, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=obj[0].keys(), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(obj)