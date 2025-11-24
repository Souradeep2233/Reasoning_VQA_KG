import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"PyTorch was built with CUDA: {torch.version.cuda}")
    print(f"Found {torch.cuda.device_count()} GPU(s).")
    print(f"Current GPU Name: {torch.cuda.get_device_name(0)}")