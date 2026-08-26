import json
import math
from datasets import load_dataset

CSV_URL = "hf://datasets/ScaleAI/mhj/harmbench_behaviors.csv"

ds = load_dataset("csv", data_files=CSV_URL, split="train")

print(ds)
print(ds.column_names)

with open("mhj_conversations.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        turns = []

        for i in range(101):
            key = f"message_{i}"
            value = row.get(key)

            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue

            text = str(value).strip()
            if not text:
                continue

            # MHJ CSV appears to store conversation messages in message_0...message_100.
            # Usually these alternate human / assistant. Adjust if you inspect otherwise.
            role = "human" if i % 2 == 0 else "assistant"
            turns.append({"role": role, "content": text})

        if not turns:
            # fallback to submission_message if message_* columns are empty
            msg = str(row.get("submission_message", "")).strip()
            if msg:
                turns = [{"role": "human", "content": msg}]

        if not turns:
            continue

        out = {
            "conversation_id": str(row.get("question_id", "")),
            "turns": turns,
            "category": str(row.get("Source", "harmbench")),
            "attack_type": str(row.get("tactic", "unknown")),
            "source": "MHJ",
            "metadata": {
                "temperature": row.get("temperature"),
                "time_spent": row.get("time_spent"),
                "submission_message": row.get("submission_message"),
            },
        }

        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print("Wrote mhj_conversations.jsonl")
