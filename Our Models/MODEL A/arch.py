import torch
import torch.nn as nn
import clip
from PIL import Image

print("--- Building Architecture with REAL CLIP Embeddings ---")

# 0. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load CLIP Model
# This will download the model if you don't have it cached
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model (ViT-B/32) loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    print("Please make sure you have 'clip' installed: pip install git+https://github.com/openai/CLIP.git")
    exit()


# 2. Define Hyperparameters
b = 4           # Image batch size
n = 10          # Number of text features (e.g., text prompts)
embed_dim = 512 # CLIP feature dimension (comes from ViT-B/32)
num_heads = 8   # Heads for cross-attention (must divide embed_dim)

print(f"Params: Image Batch (b)={b}, Text Features (n)={n}, Dim={embed_dim}\n")

# 3. Simulate INPUT Data (instead of simulating CLIP output)
# Create a batch of dummy image inputs (e.g., 4 random 224x224 images)
# We can just create a random tensor that matches the preprocessor's output shape
dummy_image_input = torch.randn(b, 3, 224, 224).to(device)

# Create a batch of dummy text inputs
dummy_text_list = [f"a prompt about {i}" for i in range(n)]
dummy_text_input = clip.tokenize(dummy_text_list).to(device)

print(f"Simulated image input shape: {dummy_image_input.shape}")
print(f"Simulated text input shape:  {dummy_text_input.shape}\n")


# 4. Get REAL CLIP Embeddings
# We use torch.no_grad() because we aren't training the CLIP model
with torch.no_grad():
    # image--->CLIP----> [b,512]
    image_features_clip = model.encode_image(dummy_image_input).float()
    # text--->CLIP----> [N,512]
    text_features_clip = model.encode_text(dummy_text_input).float()

print(f"Actual image features from CLIP: {image_features_clip.shape}")
print(f"Actual text features from CLIP:  {text_features_clip.shape}\n")


# 5. Define Your Model Components (Layers)
# linear([b,512)--->[b,512]
image_linear = nn.Linear(embed_dim, embed_dim).to(device)
# [n,512]---->[n,512]via linear
text_linear = nn.Linear(embed_dim, embed_dim).to(device)

# CA over ([b,512],[n,512])
cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)

print(f"Model Layers:")
print(f"  Image Linear: {image_linear}")
print(f"  Text Linear:  {text_linear}")
print(f"  Cross-Attention: {cross_attention}\n")


# 6. Forward Pass Simulation (using the real CLIP features)
print("--- Forward Pass ---")

# Process image features
image_feat_processed = image_linear(image_features_clip)
print(f"Processed image features: {image_feat_processed.shape}")

# Process text features
text_feat_processed = text_linear(text_features_clip)
print(f"Processed text features:  {text_feat_processed.shape}\n")

# --- Prepare for Cross-Attention ---
# Query (Q) from images: [b, 512] -> [b, 1, 512]
query = image_feat_processed.unsqueeze(1)

# Key (K) from text: [n, 512] -> [1, n, 512] -> [b, n, 512]
key = text_feat_processed.unsqueeze(0).expand(b, -1, -1)

# Value (V) from text: [n, 512] -> [1, n, 512] -> [b, n, 512]
value = text_feat_processed.unsqueeze(0).expand(b, -1, -1)

print("Preparing for Cross-Attention:")
print(f"  Query (from images): {query.shape}")
print(f"  Key (from text):   {key.shape}")
print(f"  Value (from text): {value.shape}\n")

# --- Perform Cross-Attention ---
attn_output, attn_weights = cross_attention(query, key, value)

print(f"Cross-Attention output:   {attn_output.shape}")
print(f"Cross-Attention weights: {attn_weights.shape}  (b, L, S) -> (b, 1, n)")

# Final output embedding
final_image_embedding = attn_output.squeeze(1)

print(f"\nFinal 'refined' image embedding: {final_image_embedding.shape}")