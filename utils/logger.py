# utils/logger.py
import os, csv, json
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def append_csv(path: str, row: dict, header_order=None):
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def append_jsonl(path: str, row: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def timestamp_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"