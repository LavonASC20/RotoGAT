import os
import torch

# -----------------------------------------------------------------------------
# Global Configuration for GAT-SGG Project
# -----------------------------------------------------------------------------

DATA_DIR = os.getenv("DATA_DIR", "/path/to/vqa/dataset")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Device (CUDA if available)
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", 10))  # batches
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", 1))  # epochs

# Equivariant CNN (G-CNN) Settings
GCNN_GROUP = os.getenv("GCNN_GROUP", "FlipRot2dOnR2")
GCNN_N = int(os.getenv("GCNN_N", 8))

# Region Proposal Network (RPN) Settings
RPN_PRE_NMS_TOP_N = int(os.getenv("RPN_PRE_NMS_TOP_N", 2000))
RPN_POST_NMS_TOP_N = int(os.getenv("RPN_POST_NMS_TOP_N", 1000))
RPN_NMS_THRESH = float(os.getenv("RPN_NMS_THRESH", 0.7))

# RoI Align Settings
_roi_size = os.getenv("ROI_OUTPUT_SIZE", "7,7").split(",")
ROI_OUTPUT_SIZE = (int(_roi_size[0]), int(_roi_size[1]))
ROI_SAMPLING_RATIO = int(os.getenv("ROI_SAMPLING_RATIO", 2))

# Graph & GAT Settings
NODE_EMBED_DIM = int(os.getenv("NODE_EMBED_DIM", 256))
EDGE_EMBED_DIM = int(os.getenv("EDGE_EMBED_DIM", 256))
GAT_NUM_LAYERS = int(os.getenv("GAT_NUM_LAYERS", 2))
GAT_HIDDEN_DIM = int(os.getenv("GAT_HIDDEN_DIM", 256))
GRAPH_POOL_RATIO = float(os.getenv("GRAPH_POOL_RATIO", 0.5))  
TOP_N_RELATIONS = int(os.getenv("TOP_N_RELATIONS", 50))

# Utility Settings
SEED = int(os.getenv("SEED", 42))

# Ensure reproducibility
torch.manual_seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)