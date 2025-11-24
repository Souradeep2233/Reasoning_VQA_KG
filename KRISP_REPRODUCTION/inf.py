import torch
from mmf.models.krisp import KRISP
from mmf.utils.build import build_model_from_opts
from mmf.utils.env import setup_imports
from omegaconf import OmegaConf

# 1. Load KRISP config
opts = OmegaConf.load("mmf/projects/krisp/configs/krisp/okvqa/train_val.yaml")

setup_imports()   # IMPORTANT

# 2. Build empty model
model = build_model_from_opts(opts)
model.eval()

# 3. Load checkpoint
ckpt = torch.load("/data1/souradeepd/Krisp/mmf/save/krisp_final.pth", map_location="cpu")

# 4. Load weights
model.load_state_dict(ckpt["model"])

print(model)
