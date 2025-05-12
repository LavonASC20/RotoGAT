import json
from pathlib import Path
from PIL import Image
import torch
import os
from torch.utils.data import Dataset

class SGGDataset(Dataset):
    """
    Scene Graph Generation Dataset.

    Each entry in annotations.json must have:
      - file_name:    absolute path to the image
      - boxes:        list of [x1, y1, x2, y2]
      - labels:       list of label-strings
      - relations:    list of [subj_idx, obj_idx, predicate-string]
    """
    def __init__(self, annotations_file, object2idx, rel2idx, transforms=None):
        # annotations_file: e.g. "data/annotations.json"
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        orig = len(self.annotations)
        self.annotations = [
            ann for ann in self.annotations
            if os.path.exists(ann["file_name"])
            and 
            len(ann['boxes']) > 0
        ]
        dropped = orig - len(self.annotations)
        if dropped > 0:
            print(f"SGGDataset: dropped {dropped} entries with missing images")
        self.transforms = transforms
        self.object2idx = object2idx
        self.rel2idx    = rel2idx
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(ann["file_name"]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        boxes  = torch.tensor(ann["boxes"], dtype=torch.float32)
        labels    = ann["labels"]
        relations = ann["relations"]

        target = {
            "boxes":     boxes,
            "labels":    labels,
            "relations": relations
        }

        return img, target

    def collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)

        all_targets = []
        for t in targets:
            label_ids = torch.tensor(
                [self.object2idx.get(lbl, self.object2idx["__UNK__"])
                 for lbl in t["labels"]],
                dtype=torch.long
            )

            #  each rel is [subj_idx, obj_idx, pred_str]
            edge_index = [[], []]
            rel_labels = []
            for subj_i, obj_i, pred_str in t["relations"]:
                edge_index[0].append(subj_i)
                edge_index[1].append(obj_i)
                rel_labels.append(self.rel2idx[pred_str])

            edge_index = torch.tensor(edge_index, dtype=torch.long)      
            rel_labels = torch.tensor(rel_labels, dtype=torch.long)      

            all_targets.append({
                "boxes":      t["boxes"],        
                "labels":     label_ids,         
                "edge_index": edge_index,        
                "rel_labels": rel_labels        
            })

        return images, all_targets
