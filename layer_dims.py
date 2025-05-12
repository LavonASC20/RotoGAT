from model.gat_model import SceneGraphEquiModel
from utils.vocab import build_vocabs
import torch

# build the vocabs to get class counts
obj2idx, _, rel2idx, _ = build_vocabs("data/annotations.json")
model = SceneGraphEquiModel(
    num_obj_classes = len(obj2idx),
    num_rel_classes = len(rel2idx),
    N = 8
)

print("=== R2Conv layers ===")
for name, module in model.named_modules():
    if name.startswith("conv") and hasattr(module, "filter"):
        # R2Conv stores its learnable weights in module.filter
        w = module.filter
        print(f"{name}.filter ->", tuple(w.shape))

print("\n=== BatchNorm / Pool / Act ===")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d) or \
       module.__class__.__name__.endswith("Pool") or \
       module.__class__.__name__.endswith("ReLU"):
        print(f"{name} -> {module.__class__.__name__}")

print("\n=== GAT layers & heads ===")
for name, param in model.named_parameters():
    if name.startswith("gat"):
        print(name, tuple(param.shape))
    elif name.startswith("obj_head") or name.startswith("rel_head"):
        print(name, tuple(param.shape))
