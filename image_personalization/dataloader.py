from typing import Dict, Any, Iterable, List
import gzip, json

from .conditioning import extract_titles_from_his_interaction, build_preference_text, fuse_instruction_and_preference


def jsonl_gz_reader(path: str) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_batches(data_path: str, batch_size: int = 2, adaptive_weight: float = 1.0) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for record in jsonl_gz_reader(data_path):
        instruction = record.get("instruction", "")
        his_interaction = record.get("his_interaction", "")
        item_features = record.get("item_features", "")

        titles = extract_titles_from_his_interaction(his_interaction)
        pref_text = build_preference_text(titles)
        fused_text = fuse_instruction_and_preference(instruction, pref_text, adaptive_weight=adaptive_weight)

        sample = {
            "instruction": instruction,
            "fused_preference_text": pref_text,  # To be concatenated into prompt by trainer
            "topk_desc": item_features,  # Text for semantic accuracy supervision
            "raw": record,
        }
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


