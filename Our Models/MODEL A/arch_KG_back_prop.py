import torch
import torch.nn as nn
import torch.optim as optim
import clip
from transformers import BertForMaskedLM, BertConfig, BertTokenizer 
from data import get_vqa_loader, ConceptNetFeatureExtractor
from peft import LoraConfig, get_peft_model
import numpy as np

# --- 0. Setup and Hyperparameters ---
CONCEPTNET_EMBEDDING_PATH = "/home/souradeepd/Desktop/New_arch_KIT/numberbatch-en-19.08.txt"
TRAIN_SPLIT = "validation[100:200]"
VAL_SPLIT = "validation[:50]"
TEST_SPLIT = "validation[50:100]"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# IMPROVED: Adjusted hyperparameters
batch_size = 16  # Reduced for stability
clip_embed_dim = 512
num_heads = 8    # Reduced heads
bert_model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(bert_model_name)
bert_hidden_size = config.hidden_size
bert_vocab_size = config.vocab_size

# IMPROVED: Training parameters
num_epochs = 30      # Reduced epochs
learning_rate = 5e-4 # Lower learning rate
weight_decay = 1e-4  # Added weight decay

# IMPROVED: Add learning rate scheduler
warmup_epochs = 3

# --- 1. Load Pre-trained Models ---
try:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model (ViT-B/32) loaded.")
    clip_model.requires_grad_(False)
    
    bert_for_mlm = BertForMaskedLM.from_pretrained(bert_model_name)
    language_head = bert_for_mlm.cls.to(device)
    print(f"Loaded pre-trained BERT LM Head ({bert_model_name}).")
    del bert_for_mlm
    
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    print(f"Loaded BERT Tokenizer ({bert_model_name}).")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- 1.5 Load ConceptNet Feature Extractor ---
print("\n--- Initializing ConceptNet Feature Extractor ---")
try:
    feature_extractor = ConceptNetFeatureExtractor(CONCEPTNET_EMBEDDING_PATH)
    cn_embed_dim = feature_extractor.embed_dim
    print(f"ConceptNet features loaded. Embedding dim: {cn_embed_dim}")
except FileNotFoundError:
    print(f"Error: ConceptNet embedding file not found at {CONCEPTNET_EMBEDDING_PATH}")
    exit()
except Exception as e:
    print(f"Error loading ConceptNet (did you pip install gensim spacy?): {e}")
    exit()

# --- 2. IMPROVED: Define the Complete Model Class ---
class ImprovedModel(nn.Module):
    def __init__(self, clip_model, language_head, cn_embed_dim):
        super().__init__()
        self.clip_model = clip_model
        self.language_head = language_head
        
        # IMPROVED: Better initialization and layer structure
        self.image_linear = nn.Sequential(
            nn.Linear(clip_embed_dim, clip_embed_dim),
            nn.LayerNorm(clip_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.text_linear = nn.Sequential(
            nn.Linear(clip_embed_dim, clip_embed_dim),
            nn.LayerNorm(clip_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # IMPROVED: Better cross-attention setup
        self.cross_attention = nn.MultiheadAttention(
            clip_embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # IMPROVED: Better fusion mechanism
        self.fusion_layers = nn.Sequential(
            nn.Linear(clip_embed_dim + cn_embed_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # IMPROVED: Better projection to BERT space
        self.projection_to_bert = nn.Sequential(
            nn.Linear(512, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.Dropout(0.1)
        )
        
        # IMPROVED: Add initialization
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_input, text_input, conceptnet_input):
        with torch.no_grad():
            # IMPROVED: Normalize CLIP features
            image_features_clip = self.clip_model.encode_image(image_input).float()
            text_features_clip = self.clip_model.encode_text(text_input).float()
            
            # Normalize features
            image_features_clip = torch.nn.functional.normalize(image_features_clip, p=2, dim=1)
            text_features_clip = torch.nn.functional.normalize(text_features_clip, p=2, dim=1)

        image_feat_processed = self.image_linear(image_features_clip)
        text_feat_processed = self.text_linear(text_features_clip)
        
        # IMPROVED: Better attention mechanism
        query = image_feat_processed.unsqueeze(1)
        key = text_feat_processed.unsqueeze(1)
        value = text_feat_processed.unsqueeze(1)
        
        attn_output, _ = self.cross_attention(query, key, value)
        final_image_embedding = attn_output.squeeze(1)
        
        # IMPROVED: Better fusion
        concatenated_features = torch.cat([final_image_embedding, conceptnet_input], dim=1)
        fused_features = self.fusion_layers(concatenated_features)
        bert_ready_features = self.projection_to_bert(fused_features)
        
        # IMPROVED: Add sequence dimension properly
        bert_ready_seq = bert_ready_features.unsqueeze(1)
        logits = self.language_head(bert_ready_seq)
        
        return logits.squeeze(1)

# --- 3. IMPROVED: Model Instantiation and Training Setup ---
print("\n--- Initializing Improved Model for Training ---")
model = ImprovedModel(clip_model, language_head, cn_embed_dim).to(device)

print("--- Applying LoRA ---")
lora_config = LoraConfig(
    r=16,  # Increased rank for better adaptation
    lora_alpha=32,
    target_modules=[
        "image_linear.0",  # Updated for sequential layers
        "text_linear.0",
        "fusion_layers.0",
        "fusion_layers.4",
        "projection_to_bert.0"
    ],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=[
        "cross_attention", 
        "language_head"
    ]
)

model = get_peft_model(model, lora_config)

print("Trainable parameters summary:")
model.print_trainable_parameters()

# IMPROVED: Better optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)

# IMPROVED: Add learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader) if 'train_loader' in locals() else 100,
    pct_start=0.1
)

criterion = nn.CrossEntropyLoss()

print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Scheduler: {scheduler.__class__.__name__}")
print(f"Loss Function: {criterion.__class__.__name__}\n")

# --- 4. Create all DataLoaders ---
print("--- Loading Datasets ---")
train_loader = get_vqa_loader(
    batch_size=batch_size,
    clip_preprocess=preprocess,
    bert_tokenizer=bert_tokenizer,
    feature_extractor=feature_extractor,
    split=TRAIN_SPLIT,
    shuffle=True
)
val_loader = get_vqa_loader(
    batch_size=batch_size,
    clip_preprocess=preprocess,
    bert_tokenizer=bert_tokenizer,
    feature_extractor=feature_extractor,
    split=VAL_SPLIT,
    shuffle=False
)
test_loader = get_vqa_loader(
    batch_size=batch_size,
    clip_preprocess=preprocess,
    bert_tokenizer=bert_tokenizer,
    feature_extractor=feature_extractor,
    split=TEST_SPLIT,
    shuffle=False
)
print("--- All DataLoaders Created ---")

# IMPROVED: Add gradient clipping
max_grad_norm = 1.0

# --- 5. IMPROVED: Training Loop with Better Monitoring ---
print(f"--- Starting Training for {num_epochs} Epochs ---")
best_val_acc = 0.0
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    
    # --- 5.1 Training Phase ---
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0
    
    print(f"\n--- Epoch {epoch+1:02d}/{num_epochs:02d} ---")
    print("--- Training ---")
    
    for batch_idx, batch in enumerate(train_loader):
        images, questions, cn_vectors, labels = batch
        images = images.to(device)
        questions = questions.to(device)
        cn_vectors = cn_vectors.to(device).float() 
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images, questions, cn_vectors)
        loss = criterion(logits, labels)
        loss.backward()
        
        # IMPROVED: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # IMPROVED: Better accuracy calculation
        _, predicted = torch.max(logits, 1)
        correct_in_batch = (predicted == labels).sum().item()
        total_train_correct += correct_in_batch
        total_train_samples += labels.size(0)
        batch_acc = (correct_in_batch / labels.size(0)) * 100
        
        total_train_loss += loss.item()
        
        if batch_idx % 5 == 0:  # Reduced logging frequency
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:02d} | Batch {batch_idx+1:03d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}% | LR: {current_lr:.2e}")
    
    avg_train_loss = total_train_loss / len(train_loader)
    epoch_train_acc = (total_train_correct / total_train_samples) * 100
    
    # --- 5.2 Validation Phase ---
    print("--- Running Validation ---")
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, questions, cn_vectors, labels = batch
            images = images.to(device)
            questions = questions.to(device)
            cn_vectors = cn_vectors.to(device).float() 
            labels = labels.to(device)
            
            logits = model(images, questions, cn_vectors)
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, 1)
            total_val_correct += (predicted == labels).sum().item()
            total_val_samples += labels.size(0)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    epoch_val_acc = (total_val_correct / total_val_samples) * 100
    
    # IMPROVED: Early stopping
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"*** New best model saved with val_acc: {best_val_acc:.2f}% ***")
    else:
        patience_counter += 1
    
    print(f"\n--- Epoch {epoch+1:02d} SUMMARY ---")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    print(f"Val. Loss:  {avg_val_loss:.4f} | Val. Acc:  {epoch_val_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}% | Patience: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

# --- 6. IMPROVED: Final Test with Best Model ---
print("\n--- Loading Best Model for Testing ---")
model.load_state_dict(torch.load('best_model.pth'))

print("--- Running Final Test ---")
model.eval()
total_test_correct = 0
total_test_samples = 0
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images, questions, cn_vectors, labels = batch
        images = images.to(device)
        questions = questions.to(device)
        cn_vectors = cn_vectors.to(device).float() 
        labels = labels.to(device)
        
        logits = model(images, questions, cn_vectors)
        
        _, predicted = torch.max(logits, 1)
        total_test_correct += (predicted == labels).sum().item()
        total_test_samples += labels.size(0)
        
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())
        
        print(f"Test Batch {batch_idx+1}/{len(test_loader)}")

final_test_acc = (total_test_correct / total_test_samples) * 100

# IMPROVED: Additional metrics
test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)
class_accuracy = {}
for class_id in np.unique(test_targets):
    class_mask = test_targets == class_id
    if class_mask.sum() > 0:
        class_accuracy[class_id] = (test_predictions[class_mask] == test_targets[class_mask]).mean() * 100

print("\n--- FINAL TEST RESULTS ---")
print(f"Test Accuracy: {final_test_acc:.2f}%")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print("\n--- Class-wise Accuracy (Sample) ---")
for i, (class_id, acc) in enumerate(list(class_accuracy.items())[:10]):
    print(f"Class {class_id}: {acc:.2f}%")
print("--- All Done ---")