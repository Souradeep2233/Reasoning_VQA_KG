import torch
import torch.nn as nn
import torch.optim as optim
import clip
from transformers import BertForMaskedLM, BertConfig, BertTokenizer 
from data import get_vqa_loader, ConceptNetFeatureExtractor
# NEW: Add peft imports
from peft import LoraConfig, get_peft_model

# --- 0. Setup and Hyperparameters ---
# (Unchanged)
CONCEPTNET_EMBEDDING_PATH = "/home/souradeepd/Desktop/New_arch_KIT/numberbatch-en-19.08.txt"
TRAIN_SPLIT = "validation[100:200]"
VAL_SPLIT = "validation[:50]"
TEST_SPLIT = "validation[50:100]"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
batch_size = 4      
clip_embed_dim = 512
num_heads = 8
bert_model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(bert_model_name)
bert_hidden_size = config.hidden_size
bert_vocab_size = config.vocab_size
num_epochs = 10
learning_rate = 1e-5

# --- 1. Load Pre-trained Models ---
# (Unchanged)
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
# (Unchanged)
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


# --- 2. Define the Complete Model Class ---
# (Unchanged)
class MyModel(nn.Module):
    def __init__(self, clip_model, language_head, cn_embed_dim):
        super().__init__()
        self.clip_model = clip_model
        self.language_head = language_head
        self.image_linear = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.text_linear = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            clip_embed_dim, num_heads, batch_first=True
        )
        self.projection_head = nn.Linear(clip_embed_dim + cn_embed_dim, clip_embed_dim)
        self.projection_to_bert = nn.Linear(clip_embed_dim, bert_hidden_size)

    def forward(self, image_input, text_input, conceptnet_input):
        with torch.no_grad():
            image_features_clip = self.clip_model.encode_image(image_input).float()
            text_features_clip = self.clip_model.encode_text(text_input).float()

        image_feat_processed = self.image_linear(image_features_clip)
        text_feat_processed = self.text_linear(text_features_clip)
        query = image_feat_processed.unsqueeze(1)
        key = text_feat_processed.unsqueeze(1)
        value = text_feat_processed.unsqueeze(1)
        attn_output, _ = self.cross_attention(query, key, value)
        final_image_embedding = attn_output.squeeze(1)
        concatenated_features = torch.cat([final_image_embedding, conceptnet_input], dim=1)
        projected_features = self.projection_head(concatenated_features)
        bert_ready_features = self.projection_to_bert(projected_features)
        bert_ready_seq = bert_ready_features.unsqueeze(1)
        logits = self.language_head(bert_ready_seq)
        return logits.squeeze(1)

# --- 3. MODIFIED: Instantiate Model, Apply LoRA, and Set Up Optimizer ---
print("\n--- Initializing Model for Training ---")
model = MyModel(clip_model, language_head, cn_embed_dim).to(device)

print("--- Applying LoRA ---")
# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices.
    lora_alpha=16, # Alpha parameter for scaling.
    # We target all the nn.Linear layers *we* defined in MyModel
    target_modules=[
        "image_linear",
        "text_linear",
        "projection_head",
        "projection_to_bert"
    ],
    lora_dropout=0.1,
    bias="none", # We will not train bias parameters
    # We exempt the MHA and BERT head from being frozen.
    # They will be fully fine-tuned.
    modules_to_save=[
        "cross_attention", 
        "language_head"
    ]
)

# Apply the LoRA wrapper to our model
model = get_peft_model(model, lora_config)

# NEW: Print a summary of trainable parameters
print("Trainable parameters summary:")
model.print_trainable_parameters()

# Set up the optimizer.
# model.parameters() will now *only* return the trainable parameters
# (the LoRA matrices A/B and the weights of "modules_to_save").
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Loss Function: {criterion.__class__.__name__}\n")

# --- 4. Create all DataLoaders ---
# (Unchanged)
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


# --- 5. The REAL Training Loop (with Validation) ---
# (Unchanged)
print(f"--- Starting Training for {num_epochs} Epochs ---")
for epoch in range(num_epochs):
    
    # --- 5.1 Training Phase ---
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0
    num_train_batches = 0
    
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
        optimizer.step()
        
        _, predicted = torch.max(logits, 1)
        correct_in_batch = (predicted == labels).sum().item()
        total_train_correct += correct_in_batch
        total_train_samples += labels.size(0)
        batch_acc = (correct_in_batch / labels.size(0)) * 100
        
        total_train_loss += loss.item()
        num_train_batches += 1
        
        print(f"Epoch {epoch+1:02d} | Train Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.2f}%")
    
    avg_train_loss = total_train_loss / num_train_batches
    epoch_train_acc = (total_train_correct / total_train_samples) * 100
    
    # --- 5.2 Validation Phase ---
    print("--- Running Validation ---")
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    num_val_batches = 0
    
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
            num_val_batches += 1
            
            print(f"Epoch {epoch+1:02d} | Validation Batch {batch_idx+1}/{len(val_loader)}")

    avg_val_loss = total_val_loss / num_val_batches
    epoch_val_acc = (total_val_correct / total_val_samples) * 100
    
    print(f"\n--- Epoch {epoch+1:02d} SUMMARY ---")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    print(f"Val. Loss:  {avg_val_loss:.4f} | Val. Acc:  {epoch_val_acc:.2f}%")


# --- 6. Final Test Loop ---
# (Unchanged)
print("\n--- Training Finished. Running Final Test ---")
model.eval()
total_test_loss = 0
total_test_correct = 0
total_test_samples = 0
num_test_batches = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images, questions, cn_vectors, labels = batch
        images = images.to(device)
        questions = questions.to(device)
        cn_vectors = cn_vectors.to(device).float() 
        labels = labels.to(device)
        
        logits = model(images, questions, cn_vectors)
        loss = criterion(logits, labels)
        
        _, predicted = torch.max(logits, 1)
        total_test_correct += (predicted == labels).sum().item()
        total_test_samples += labels.size(0)
        
        total_test_loss += loss.item()
        num_test_batches += 1
        
        print(f"Test Batch {batch_idx+1}/{len(test_loader)}")

avg_test_loss = total_test_loss / num_test_batches
final_test_acc = (total_test_correct / total_test_samples) * 100

print("\n--- FINAL TEST RESULTS ---")
print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {final_test_acc:.2f}%")
print("--- All Done ---")