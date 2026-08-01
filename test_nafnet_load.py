import torch

ckpt_path = "checkpoints/nafnet_gopro.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Top-level keys in checkpoint:", list(ckpt.keys())[:5])

if "params" in ckpt:
    state_dict = ckpt["params"]
elif "params_ema" in ckpt:
    state_dict = ckpt["params_ema"]
else:
    state_dict = ckpt

print("\nNumber of weight tensors:", len(state_dict))
print("\nFirst 10 layer names:")
for i, k in enumerate(state_dict.keys()):
    if i >= 10:
        break
    print(" ", k)