import os
import argparse
import glob
import math                                                   
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import rotate          

from data.dataset import SGGDataset
from utils.vocab import build_vocabs
from model.gat_model import SceneGraphEquiModel
import matplotlib.pyplot as plt

def collate_for_eval(batch):
    return batch[0][0], batch[0][1]

def rotate_boxes(boxes, angle, image_size):
    H, W = image_size
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    R = torch.tensor([[ cos, -sin],
                      [ sin,  cos]], device=boxes.device)

    x1,y1,x2,y2 = boxes.unbind(1)
    corners = torch.stack([
        torch.stack([x1,y1], dim=1),
        torch.stack([x1,y2], dim=1),
        torch.stack([x2,y1], dim=1),
        torch.stack([x2,y2], dim=1),
    ], dim=1)  

    center = torch.tensor([W/2, H/2], device=boxes.device)
    corners = (corners - center) @ R.T + center

    x_coords = corners[...,0]
    y_coords = corners[...,1]
    new_x1 = x_coords.min(dim=1).values
    new_y1 = y_coords.min(dim=1).values
    new_x2 = x_coords.max(dim=1).values
    new_y2 = y_coords.max(dim=1).values

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="path to annotations.json")
    parser.add_argument("--checkpoint", required=True,
                        help="which .pt checkpoint to load")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--N", type=int, default=8,
                        help="number of discrete rotations in backbone")
    parser.add_argument("--data-root", default="data",
                        help="root folder containing images/")
    args = parser.parse_args()

    object2idx, idx2object, rel2idx, idx2rel = build_vocabs(args.annotations)
    num_obj = len(idx2object)
    num_rel = len(idx2rel)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = SceneGraphEquiModel(
        num_obj_classes=num_obj,
        num_rel_classes=num_rel,
        N=args.N
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"Missing keys (will use random init): {missing}")
    if unexpected:
        print(f"Unexpected keys (in checkpoint but not in model): {unexpected}")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transforms = Compose([ Resize((600,800)), ToTensor() ])
    ds = SGGDataset(
        annotations_file=args.annotations,
        transforms=transforms,
        object2idx=object2idx,
        rel2idx=rel2idx
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ds.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    ANGLES = [i * (360.0 / args.N) for i in range(args.N)]
    recall_by_angle = {angle: {20:0, 50:0, 100:0} for angle in ANGLES}
    total_gt_rels = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            all_rois   = [t["boxes"].to(device) for t in targets]
            edge_index = torch.cat(
                [t["edge_index"] + offset
                 for offset, t in enumerate(targets)],
                dim=1
            ).to(device)

            gt_edge_index = targets[0]["edge_index"]
            gt_rel_labels = targets[0]["rel_labels"]
            E_gt = gt_rel_labels.numel()
            total_gt_rels += E_gt
            gt_triplets = set(zip(
                gt_edge_index[0].tolist(),
                gt_edge_index[1].tolist(),
                gt_rel_labels.tolist()
            ))

            for angle in ANGLES:
                img_rot = rotate(images, angle)
                rois_rot = [ rotate_boxes(r, angle, img_rot.shape[-2:]) for r in all_rois ]

                obj_logits, rel_logits = model(img_rot, rois_rot, edge_index)

                scores, pred_labels = rel_logits.max(dim=1)
                subj_idx = edge_index[0].tolist()
                obj_idx  = edge_index[1].tolist()
                pred_triplets = [
                    (subj_idx[i], obj_idx[i], pred_labels[i].item(), scores[i].item())
                    for i in range(len(scores))
                ]
                pred_triplets.sort(key=lambda x: x[3], reverse=True)

                for K in recall_by_angle[angle]:
                    topk = pred_triplets[:K]
                    correct = sum(1 for (s,o,r,_) in topk if (s,o,r) in gt_triplets)
                    recall_by_angle[angle][K] += correct

    print(f"Total ground-truth relations: {total_gt_rels}\n")
    for angle in sorted(ANGLES):
        print(f"=== Angle {angle:.1f}° ===")
        for K in sorted(recall_by_angle[angle]):
            R = recall_by_angle[angle][K] / total_gt_rels * 100.0
            print(f"  Recall@{K}: {recall_by_angle[angle][K]}/{total_gt_rels} = {R:.2f}%")
        print()

if __name__=="__main__":
    main()
