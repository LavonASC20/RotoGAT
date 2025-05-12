import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from data.dataset import SGGDataset
from utils.vocab import build_vocabs
from model.gat_model import SceneGraphEquiModel

def main():
    DATA_ROOT  = "data"
    ANNOTATIONS = os.path.join(DATA_ROOT, "annotations.json")
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    NUM_SMALL = 10000
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    object2idx, idx2object, rel2idx, idx2rel = build_vocabs(ANNOTATIONS)
    num_obj_classes = len(idx2object)
    num_rel_classes = len(idx2rel)

    dataset = SGGDataset(
        annotations_file=ANNOTATIONS,
        transforms=Compose([
            Resize((600, 800)),   
            ToTensor(),
        ]),
        object2idx=object2idx,
        rel2idx=rel2idx
    )  

    small_dataset = Subset(dataset, list(range(min(NUM_SMALL, len(dataset)))))
    loader = DataLoader(
        small_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = SceneGraphEquiModel(
        num_obj_classes=num_obj_classes,
        num_rel_classes=num_rel_classes,
        N=8                # 8-fold rotations
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    obj_loss_fn = nn.CrossEntropyLoss()
    rel_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_obj_loss = 0.0
        running_rel_loss = 0.0
        obj_loss_history = []
        rel_loss_history = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for images, targets in pbar:
            images = images.to(DEVICE)
    
            batch_boxes     = [t["boxes"].to(DEVICE) for t in targets]
            batch_labels    = torch.cat([t["labels"].to(DEVICE) for t in targets], dim=0)
            edge_index      = torch.cat([t["edge_index"] + offset
                                        for offset, t in enumerate(targets)], dim=1).to(DEVICE)
            rel_labels      = torch.cat([t["rel_labels"].to(DEVICE) for t in targets], dim=0)

            optimizer.zero_grad()

            all_rois = []
            offset = 0
            for batch_idx, t in enumerate(targets):
                all_rois.append(t["boxes"])

            obj_logits, rel_logits = model(images, all_rois, edge_index)
            if rel_labels.numel() == 0:
                loss_rel = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            else:
                loss_rel = rel_loss_fn(rel_logits, rel_labels)

            loss_obj = obj_loss_fn(obj_logits, batch_labels)
            loss_rel = rel_loss_fn(rel_logits, rel_labels)
            loss = loss_obj + loss_rel
            loss.backward()
            optimizer.step()

            running_obj_loss += loss_obj.item()
            running_rel_loss += loss_rel.item()

            denom = max(1, pbar.n)
            pbar.set_postfix({
                "L_obj": f"{running_obj_loss / denom:.4f}",
                "L_rel": f"{running_rel_loss / denom:.4f}"
            })
            avg_obj = running_obj_loss / len(loader)
            avg_rel = running_rel_loss / len(loader)
            obj_loss_history.append(avg_obj)
            rel_loss_history.append(avg_rel)

        ckpt = {
            "model_state":    model.state_dict(),
            "opt_state":      optimizer.state_dict(),
            "epoch":          epoch,
            "object2idx":     object2idx,
            "rel2idx":        rel2idx,
            "obj_loss":       avg_obj,
            "rel_loss":       avg_rel,
        }
        torch.save(ckpt, f"checkpoint_epoch{epoch}.pt")
        print(f"\n Saved checkpoint_epoch{epoch}.pt\n")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()