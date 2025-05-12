import json
from pathlib import Path

def build_object_vocab(annotations_path):
    annotations_path = Path(annotations_path)
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    labels = set()
    for entry in annotations:
        labels.update(entry.get("labels", []))

    idx2object = ["__UNK__"] + sorted(labels)
    object2idx = { label: idx for idx, label in enumerate(idx2object) }
    return object2idx, idx2object

def build_relation_vocab(annotations_path):
    annotations_path = Path(annotations_path)
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    preds = set()
    for entry in annotations:
        for subj_i, obj_i, pred in entry.get("relations", []):
            preds.add(pred)

    idx2rel = sorted(preds)
    rel2idx = { rel: idx for idx, rel in enumerate(idx2rel) }
    return rel2idx, idx2rel

def build_vocabs(annotations_path):
    object2idx, idx2object = build_object_vocab(annotations_path)
    rel2idx,    idx2rel    = build_relation_vocab(annotations_path)
    return object2idx, idx2object, rel2idx, idx2rel
