import torch
import torch.nn as nn
import clip
from transformers import BertForMaskedLM, BertConfig

print("--- Building Architecture with CLIP + ConceptNet + BERT LM Head ---")

# 0. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load CLIP Model
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model (ViT-B/32) loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit()


# 2. Define Hyperparameters
b = 4           # Image batch size
n = 10          # Number of text features
clip_embed_dim = 512 # CLIP feature dimension
cn_embed_dim = 300   # ConceptNet embedding dimension
num_heads = 8

# BERT-specific params
bert_model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(bert_model_name)
bert_hidden_size = config.hidden_size # This is 768
bert_vocab_size = config.vocab_size   # This is 30522

print(f"Params: Image Batch (b)={b}, Text Features (n)={n}")
print(f"BERT Head: Using '{bert_model_name}' (Hidden: {bert_hidden_size}, Vocab: {bert_vocab_size})\n")


# 3. Simulate INPUT Data
dummy_image_input = torch.randn(b, 3, 224, 224).to(device)
dummy_text_list = [f"a prompt about {i}" for i in range(n)]
dummy_text_input = clip.tokenize(dummy_text_list).to(device)


# 4. Get REAL CLIP Embeddings
with torch.no_grad():
    image_features_clip = model.encode_image(dummy_image_input).float()
    text_features_clip = model.encode_text(dummy_text_input).float()

print(f"Actual image features from CLIP: {image_features_clip.shape}")
print(f"Actual text features from CLIP:  {text_features_clip.shape}\n")


# 5. Define Your Model Components (Layers)
print(f"--- Model Layers Definition ---")
# --- CLIP/Attention Layers ---
image_linear = nn.Linear(clip_embed_dim, clip_embed_dim).to(device)
text_linear = nn.Linear(clip_embed_dim, clip_embed_dim).to(device)
cross_attention = nn.MultiheadAttention(clip_embed_dim, num_heads, batch_first=True).to(device)

# --- ConceptNet Fusion Layers ---
# Projects [b, 512+300] -> [b, 512]
projection_head = nn.Linear(clip_embed_dim + cn_embed_dim, clip_embed_dim).to(device)
print(f"  Projection Head (812 -> 512): {projection_head}")

# --- BERT Language Head ---
# New layer to project our 512 dim to BERT's 768 dim
projection_to_bert = nn.Linear(clip_embed_dim, bert_hidden_size).to(device)
print(f"  Projection to BERT (512 -> {bert_hidden_size}): {projection_to_bert}")

# Load pre-trained BERT and grab *only* its Masked LM head
try:
    bert_for_mlm = BertForMaskedLM.from_pretrained(bert_model_name)
    language_head = bert_for_mlm.cls.to(device)
    print(f"  Loaded pre-trained BERT LM Head ({bert_model_name})")
    
    # We don't need the full BERT model, just the head
    del bert_for_mlm 
except Exception as e:
    print(f"Error loading transformers model: {e}")
    print("Please make sure you have 'transformers' installed: pip install transformers")
    exit()

print(f"  Language Head Layers: {language_head}\n")


# 6. Forward Pass (Cross-Attention)
print("--- Forward Pass ---")
print("Step 6: Cross-Attention")

image_feat_processed = image_linear(image_features_clip)
text_feat_processed = text_linear(text_features_clip)

query = image_feat_processed.unsqueeze(1)
key = text_feat_processed.unsqueeze(0).expand(b, -1, -1)
value = text_feat_processed.unsqueeze(0).expand(b, -1, -1)

attn_output, _ = cross_attention(query, key, value)
final_image_embedding = attn_output.squeeze(1)
print(f"  Cross-Attention output (squeezed): {final_image_embedding.shape}")


# 7. Simulate ConceptNet Embedding Lookup
print("Step 7: ConceptNet Lookup (Simulated)")
conceptnet_embedding = torch.rand(1, cn_embed_dim).to(device)
conceptnet_expanded = conceptnet_embedding.expand(b, -1)
print(f"  ConceptNet vector expanded for batch: {conceptnet_expanded.shape}")


# 8. Concatenate and Project
print("Step 8: Concatenate and Project")
concatenated_features = torch.cat([final_image_embedding, conceptnet_expanded], dim=1)
print(f"  Concatenated features (CLIP + CN): {concatenated_features.shape}")

projected_features = projection_head(concatenated_features)
print(f"  Projected features: {projected_features.shape}")


# 9. Pass through BERT Language Head
print("Step 9: BERT Language Head")
# Project from 512 to BERT's hidden size 768
bert_ready_features = projection_to_bert(projected_features)
print(f"  Projected to BERT hidden size: {bert_ready_features.shape}")

# Add a "sequence length" dimension of 1
bert_ready_seq = bert_ready_features.unsqueeze(1)
print(f"  Unsqueezed for sequence: {bert_ready_seq.shape}")

# Pass through the BERT head
logits = language_head(bert_ready_seq)
print(f"  Output from BERT head: {logits.shape}")

# Squeeze the "sequence" dimension back out
final_logits = logits.squeeze(1)
print(f"\n  Final logits output: {final_logits.shape}")
print(f"    (Batch Size, Vocab Size) = ({b}, {bert_vocab_size})")