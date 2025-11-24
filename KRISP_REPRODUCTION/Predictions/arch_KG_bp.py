import torch
import torch.nn as nn
import torch.optim as optim
import clip
from transformers import BertForMaskedLM, BertConfig

# --- 0. Setup and Hyperparameters ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model Hyperparameters
b = 4           # Image batch size
n = 10          # Number of text features
clip_embed_dim = 512
cn_embed_dim = 300
num_heads = 8
bert_model_name = 'bert-base-uncased'

# BERT config
config = BertConfig.from_pretrained(bert_model_name)
bert_hidden_size = config.hidden_size # 768
bert_vocab_size = config.vocab_size   # 30522

# Training Hyperparameters
num_epochs = 10
learning_rate = 1e-5

# --- 1. Load Pre-trained Models ---
# Load CLIP
try:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model (ViT-B/32) loaded.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit()
    
# FREEZE CLIP: This is crucial. We don't want to backprop into CLIP.
clip_model.requires_grad_(False)

# Load BERT LM Head
try:
    bert_for_mlm = BertForMaskedLM.from_pretrained(bert_model_name)
    language_head = bert_for_mlm.cls.to(device)
    print(f"Loaded pre-trained BERT LM Head ({bert_model_name}).")
    del bert_for_mlm # We only need the head
except Exception as e:
    print(f"Error loading transformers model: {e}")
    exit()

# Note: We are choosing to *fine-tune* the BERT head, so we don't freeze it.
# If you wanted to freeze it, you would add:
# language_head.requires_grad_(False)


# --- 2. Define the Complete Model Class ---
class MyModel(nn.Module):
    def __init__(self, clip_model, language_head):
        super().__init__()
        
        # Store pre-trained models
        self.clip_model = clip_model
        self.language_head = language_head

        # --- Define trainable layers ---
        self.image_linear = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.text_linear = nn.Linear(clip_embed_dim, clip_embed_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            clip_embed_dim, num_heads, batch_first=True
        )
        
        # [b, 512+300] -> [b, 512]
        self.projection_head = nn.Linear(clip_embed_dim + cn_embed_dim, clip_embed_dim)
        
        # [b, 512] -> [b, 768] (BERT's hidden size)
        self.projection_to_bert = nn.Linear(clip_embed_dim, bert_hidden_size)

    def forward(self, image_input, text_input, conceptnet_input):
        # Step 1: Get CLIP features (this part has no gradients)
        with torch.no_grad():
            image_features_clip = self.clip_model.encode_image(image_input).float()
            text_features_clip = self.clip_model.encode_text(text_input).float()

        # Step 2: Pass through linear layers (trainable)
        image_feat_processed = self.image_linear(image_features_clip)
        text_feat_processed = self.text_linear(text_features_clip)

        # Step 3: Cross-Attention (trainable)
        query = image_feat_processed.unsqueeze(1)
        key = text_feat_processed.unsqueeze(0).expand(b, -1, -1)
        value = text_feat_processed.unsqueeze(0).expand(b, -1, -1)
        attn_output, _ = self.cross_attention(query, key, value)
        final_image_embedding = attn_output.squeeze(1) # Shape: [b, 512]

        # Step 4: Concatenate with ConceptNet (trainable)
        # Expand conceptnet vector to batch size
        conceptnet_expanded = conceptnet_input.expand(b, -1)
        
        concatenated_features = torch.cat([final_image_embedding, conceptnet_expanded], dim=1)
        projected_features = self.projection_head(concatenated_features) # Shape: [b, 512]

        # Step 5: Pass to BERT Head (trainable)
        bert_ready_features = self.projection_to_bert(projected_features) # Shape: [b, 768]
        bert_ready_seq = bert_ready_features.unsqueeze(1) # Shape: [b, 1, 768]
        
        logits = self.language_head(bert_ready_seq) # Shape: [b, 1, vocab_size]
        final_logits = logits.squeeze(1) # Shape: [b, vocab_size]
        
        return final_logits

# --- 3. Instantiate Model, Optimizer, and Loss ---
print("\n--- Initializing Model for Training ---")

# Instantiate the model
model = MyModel(clip_model, language_head).to(device)

# Define Optimizer
# model.parameters() will *only* list the trainable parameters 
# (i.e., not the frozen CLIP model)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
print(f"Optimizer: {optimizer.__class__.__name__}")

# Define Loss Function
criterion = nn.CrossEntropyLoss()
print(f"Loss Function: {criterion.__class__.__name__}\n")


# --- 4. The Training Loop ---
print(f"--- Starting Training for {num_epochs} Epochs ---")

for epoch in range(num_epochs):
    
    # --- Create Dummy Data for this step ---
    # In a real loop, this data would come from your DataLoader
    dummy_image_input = torch.randn(b, 3, 224, 224).to(device)
    dummy_text_list = [f"a prompt about {i}" for i in range(n)]
    dummy_text_input = clip.tokenize(dummy_text_list).to(device)
    dummy_conceptnet_input = torch.rand(1, cn_embed_dim).to(device)
    
    # Create dummy labels (e.g., we want the model to predict token index 100 for
    # the first item, 50 for the second, etc.)
    dummy_labels = torch.randint(0, bert_vocab_size, (b,)).to(device)
    
    # --- Core Backpropagation Steps ---
    
    # 1. Zero the gradients
    optimizer.zero_grad()
    
    # 2. Forward pass: Get model predictions
    logits = model(dummy_image_input, dummy_text_input, dummy_conceptnet_input)
    
    # 3. Calculate the loss
    loss = criterion(logits, dummy_labels)
    
    # 4. Backward pass: Compute gradients
    loss.backward()
    
    # 5. Update weights
    optimizer.step()
    
    # --- End of Core Steps ---
    
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {loss.item():.4f}")

print("--- Training Finished ---")