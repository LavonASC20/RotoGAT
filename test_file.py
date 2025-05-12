from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from data.dataset import SGGDataset
from utils.vocab import build_vocabs
object2idx, idx2object, rel2idx, idx2rel = build_vocabs("data/annotations.json")


ds = SGGDataset(
    annotations_file="data/annotations.json",
    transforms=ToTensor(),      
    object2idx=object2idx,
    rel2idx=rel2idx     
)

loader = DataLoader(
    ds,
    batch_size=2,
    collate_fn=ds.collate_fn
)

imgs, targets = next(iter(loader))
print(imgs.shape, targets)
