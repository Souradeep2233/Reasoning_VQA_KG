import torch
import torch.nn as nn
import torch.optim as optim
import clip
from transformers import BertForMaskedLM, BertConfig, BertTokenizer 
from data import get_vqa_loader, ConceptNetFeatureExtractor
from peft import LoraConfig, get_peft_model
import numpy as np
import logging
import csv
import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- 0. Setup and Hyperparameters ---
CONCEPTNET_EMBEDDING_PATH = "/home/souradeepd/Desktop/New_arch_KIT/numberbatch-en-19.08.txt"
TRAIN_SPLIT = "validation[100:200]"
VAL_SPLIT = "validation[:50]"
TEST_SPLIT = "validation[50:100]"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('runs', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Setup comprehensive logging
experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# File logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/{experiment_name}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TensorBoard
writer = SummaryWriter(f'runs/{experiment_name}')

# CSV logger
csv_file = f'logs/{experiment_name}_metrics.csv'
with open(csv_file, 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['epoch', 'batch', 'phase', 'loss', 'accuracy', 'learning_rate', 'timestamp'])

logger.info(f"Using device: {device}")
logger.info(f"Experiment: {experiment_name}")

# Hyperparameters
batch_size = 16
clip_embed_dim = 512
num_heads = 8
bert_model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(bert_model_name)
bert_hidden_size = config.hidden_size
bert_vocab_size = config.vocab_size

# Training parameters
num_epochs = 30
learning_rate = 5e-4
weight_decay = 1e-4
warmup_epochs = 3

# Save hyperparameters
hyperparams = {
    'batch_size': batch_size,
    'clip_embed_dim': clip_embed_dim,
    'num_heads': num_heads,
    'bert_model_name': bert_model_name,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'num_epochs': num_epochs,
    'experiment_name': experiment_name
}

with open(f'logs/{experiment_name}_hyperparams.json', 'w') as f:
    json.dump(hyperparams, f, indent=2)

# --- 1. Load Pre-trained Models ---
try:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    logger.info("CLIP model (ViT-B/32) loaded.")
    clip_model.requires_grad_(False)
    
    bert_for_mlm = BertForMaskedLM.from_pretrained(bert_model_name)
    language_head = bert_for_mlm.cls.to(device)
    logger.info(f"Loaded pre-trained BERT LM Head ({bert_model_name}).")
    del bert_for_mlm
    
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    logger.info(f"Loaded BERT Tokenizer ({bert_model_name}).")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    exit()

# --- 1.5 Load ConceptNet Feature Extractor ---
logger.info("Initializing ConceptNet Feature Extractor")
try:
    feature_extractor = ConceptNetFeatureExtractor(CONCEPTNET_EMBEDDING_PATH)
    cn_embed_dim = feature_extractor.embed_dim
    logger.info(f"ConceptNet features loaded. Embedding dim: {cn_embed_dim}")
except FileNotFoundError:
    logger.error(f"Error: ConceptNet embedding file not found at {CONCEPTNET_EMBEDDING_PATH}")
    exit()
except Exception as e:
    logger.error(f"Error loading ConceptNet: {e}")
    exit()

# --- 2. Define the Complete Model Class ---
class ImprovedModel(nn.Module):
    def __init__(self, clip_model, language_head, cn_embed_dim):
        super().__init__()
        self.clip_model = clip_model
        self.language_head = language_head
        
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
        
        self.cross_attention = nn.MultiheadAttention(
            clip_embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        
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
        
        self.projection_to_bert = nn.Sequential(
            nn.Linear(512, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.Dropout(0.1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_input, text_input, conceptnet_input):
        with torch.no_grad():
            image_features_clip = self.clip_model.encode_image(image_input).float()
            text_features_clip = self.clip_model.encode_text(text_input).float()
            
            image_features_clip = torch.nn.functional.normalize(image_features_clip, p=2, dim=1)
            text_features_clip = torch.nn.functional.normalize(text_features_clip, p=2, dim=1)

        image_feat_processed = self.image_linear(image_features_clip)
        text_feat_processed = self.text_linear(text_features_clip)
        
        query = image_feat_processed.unsqueeze(1)
        key = text_feat_processed.unsqueeze(1)
        value = text_feat_processed.unsqueeze(1)
        
        attn_output, _ = self.cross_attention(query, key, value)
        final_image_embedding = attn_output.squeeze(1)
        
        concatenated_features = torch.cat([final_image_embedding, conceptnet_input], dim=1)
        fused_features = self.fusion_layers(concatenated_features)
        bert_ready_features = self.projection_to_bert(fused_features)
        
        bert_ready_seq = bert_ready_features.unsqueeze(1)
        logits = self.language_head(bert_ready_seq)
        
        return logits.squeeze(1)

# --- 3. Model Instantiation and Training Setup ---
logger.info("Initializing Improved Model for Training")
model = ImprovedModel(clip_model, language_head, cn_embed_dim).to(device)

logger.info("Applying LoRA")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "image_linear.0",
        "text_linear.0",
        "fusion_layers.0",
        "fusion_layers.4",
        "projection_to_bert.0"
    ],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["cross_attention", "language_head"]
)

model = get_peft_model(model, lora_config)

# Log model architecture
logger.info("Trainable parameters summary:")
model.print_trainable_parameters()

# Save model architecture
with open(f'logs/{experiment_name}_model_architecture.txt', 'w') as f:
    f.write(str(model))

# Optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)

criterion = nn.CrossEntropyLoss()

logger.info(f"Optimizer: {optimizer.__class__.__name__}")
logger.info(f"Loss Function: {criterion.__class__.__name__}")

# --- 4. Create all DataLoaders ---
logger.info("Loading Datasets")
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

# Setup scheduler after train_loader is defined
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1
)

logger.info(f"Scheduler: {scheduler.__class__.__name__}")
logger.info("All DataLoaders Created")

# --- Utility functions for logging and plotting ---
def log_metrics_to_csv(epoch, batch, phase, loss, accuracy, lr):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, batch, phase, loss, accuracy, lr, datetime.now().isoformat()])

def save_predictions(predictions, targets, probabilities, phase, epoch=None):
    filename = f'predictions/{experiment_name}_{phase}'
    if epoch is not None:
        filename += f'_epoch{epoch}'
    filename += '.npz'
    
    np.savez(filename,
             predictions=np.array(predictions),
             targets=np.array(targets),
             probabilities=np.array(probabilities),
             timestamp=datetime.now().isoformat())

def create_training_plots(epoch_data, experiment_name):
    """Create and save training plots"""
    epochs = epoch_data['epochs']
    train_losses = epoch_data['train_losses']
    val_losses = epoch_data['val_losses']
    train_accs = epoch_data['train_accs']
    val_accs = epoch_data['val_accs']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss vs Epoch
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss vs Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Epoch
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined Loss and Accuracy
    ax3.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax3.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_title('Loss Trends')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, train_accs, 'g--', linewidth=2, label='Train Acc', alpha=0.8)
    ax3_twin.plot(epochs, val_accs, 'm--', linewidth=2, label='Val Acc', alpha=0.8)
    ax3_twin.set_ylabel('Accuracy (%)', color='g')
    ax3_twin.tick_params(axis='y', labelcolor='g')
    ax3_twin.legend(loc='upper right')
    
    # Plot 4: Gap between train and val (overfitting indicator)
    loss_gap = [t - v for t, v in zip(train_losses, val_losses)]
    acc_gap = [v - t for t, v in zip(train_accs, val_accs)]
    
    ax4.plot(epochs, loss_gap, 'orange', linewidth=2, label='Loss Gap (Train - Val)', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Gap', color='orange')
    ax4.tick_params(axis='y', labelcolor='orange')
    ax4.set_title('Overfitting Indicators')
    ax4.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, acc_gap, 'purple', linewidth=2, label='Acc Gap (Val - Train)', alpha=0.8)
    ax4_twin.set_ylabel('Accuracy Gap', color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    ax4_twin.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig(f'plots/{experiment_name}_training_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'plots/{experiment_name}_training_plots.pdf', bbox_inches='tight')
    plt.close()
    
    # Create individual plots as well
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', linewidth=3, label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', linewidth=3, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Epoch', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.savefig(f'plots/{experiment_name}_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_accs, 'b-', linewidth=3, label='Training Accuracy')
    ax.plot(epochs, val_accs, 'r-', linewidth=3, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Epoch', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.savefig(f'plots/{experiment_name}_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training plots saved to plots/{experiment_name}_*.png")

# --- 5. Training Loop with Comprehensive Logging ---
max_grad_norm = 1.0
best_val_acc = 0.0
patience = 5
patience_counter = 0

# Lists to store epoch metrics for plotting
epoch_data = {
    'epochs': [],
    'train_losses': [],
    'val_losses': [],
    'train_accs': [],
    'val_accs': []
}

logger.info(f"Starting Training for {num_epochs} Epochs")

for epoch in range(num_epochs):
    
    # Training Phase
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0
    train_predictions = []
    train_targets = []
    train_probabilities = []
    
    logger.info(f"Epoch {epoch+1:02d}/{num_epochs:02d} - Training")
    
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # Calculate metrics
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, 1)
        correct_in_batch = (predicted == labels).sum().item()
        total_train_correct += correct_in_batch
        total_train_samples += labels.size(0)
        batch_acc = (correct_in_batch / labels.size(0)) * 100
        
        total_train_loss += loss.item()
        
        # Store predictions
        train_predictions.extend(predicted.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        train_probabilities.extend(probabilities.detach().cpu().numpy())
        
        current_lr = scheduler.get_last_lr()[0]
        
        # Log batch metrics
        if batch_idx % 5 == 0:
            logger.info(f"Epoch {epoch+1:02d} | Batch {batch_idx+1:03d}/{len(train_loader)} | "
                       f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}% | LR: {current_lr:.2e}")
        
        # Log to CSV and TensorBoard
        log_metrics_to_csv(epoch+1, batch_idx, 'train', loss.item(), batch_acc, current_lr)
        writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Accuracy/train_batch', batch_acc, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Learning_rate', current_lr, epoch * len(train_loader) + batch_idx)
    
    # Epoch training summary
    avg_train_loss = total_train_loss / len(train_loader)
    epoch_train_acc = (total_train_correct / total_train_samples) * 100
    
    logger.info(f"Epoch {epoch+1:02d} Training Summary - Loss: {avg_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%")
    
    # Save training predictions for this epoch
    save_predictions(train_predictions, train_targets, train_probabilities, 'train', epoch+1)
    
    # Validation Phase
    logger.info(f"Epoch {epoch+1:02d} - Validation")
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    val_predictions = []
    val_targets = []
    val_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, questions, cn_vectors, labels = batch
            images = images.to(device)
            questions = questions.to(device)
            cn_vectors = cn_vectors.to(device).float() 
            labels = labels.to(device)
            
            logits = model(images, questions, cn_vectors)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            
            _, predicted = torch.max(logits, 1)
            total_val_correct += (predicted == labels).sum().item()
            total_val_samples += labels.size(0)
            total_val_loss += loss.item()
            
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            val_probabilities.extend(probabilities.detach().cpu().numpy())
            
            batch_acc = ((predicted == labels).sum().item() / labels.size(0)) * 100
            log_metrics_to_csv(epoch+1, batch_idx, 'val', loss.item(), batch_acc, current_lr)
    
    avg_val_loss = total_val_loss / len(val_loader)
    epoch_val_acc = (total_val_correct / total_val_samples) * 100
    
    # Save validation predictions
    save_predictions(val_predictions, val_targets, val_probabilities, 'val', epoch+1)
    
    # Store metrics for plotting
    epoch_data['epochs'].append(epoch + 1)
    epoch_data['train_losses'].append(avg_train_loss)
    epoch_data['val_losses'].append(avg_val_loss)
    epoch_data['train_accs'].append(epoch_train_acc)
    epoch_data['val_accs'].append(epoch_val_acc)
    
    # Log epoch summaries
    writer.add_scalars('Loss/epoch', {
        'train': avg_train_loss,
        'val': avg_val_loss
    }, epoch)
    
    writer.add_scalars('Accuracy/epoch', {
        'train': epoch_train_acc,
        'val': epoch_val_acc
    }, epoch)
    
    # Early stopping and model saving
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': epoch_val_acc,
            'train_acc': epoch_train_acc,
            'loss': avg_val_loss,
            'epoch_data': epoch_data
        }, f'logs/{experiment_name}_best_model.pth')
        logger.info(f"*** New best model saved with val_acc: {best_val_acc:.2f}% ***")
    else:
        patience_counter += 1
    
    # Epoch summary
    logger.info(f"Epoch {epoch+1:02d} SUMMARY")
    logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    logger.info(f"Val. Loss:  {avg_val_loss:.4f} | Val. Acc:  {epoch_val_acc:.2f}%")
    logger.info(f"Best Val Acc: {best_val_acc:.2f}% | Patience: {patience_counter}/{patience}")
    
    # Create plots after each epoch
    create_training_plots(epoch_data, experiment_name)
    
    if patience_counter >= patience:
        logger.info("Early stopping triggered!")
        break

# --- 6. Final Test with Best Model ---
logger.info("Loading Best Model for Testing")
checkpoint = torch.load(f'logs/{experiment_name}_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

logger.info("Running Final Test")
model.eval()
total_test_correct = 0
total_test_samples = 0
test_predictions = []
test_targets = []
test_probabilities = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images, questions, cn_vectors, labels = batch
        images = images.to(device)
        questions = questions.to(device)
        cn_vectors = cn_vectors.to(device).float() 
        labels = labels.to(device)
        
        logits = model(images, questions, cn_vectors)
        probabilities = torch.softmax(logits, dim=1)
        
        _, predicted = torch.max(logits, 1)
        total_test_correct += (predicted == labels).sum().item()
        total_test_samples += labels.size(0)
        
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())
        test_probabilities.extend(probabilities.detach().cpu().numpy())
        
        batch_acc = ((predicted == labels).sum().item() / labels.size(0)) * 100
        logger.info(f"Test Batch {batch_idx+1}/{len(test_loader)} - Acc: {batch_acc:.2f}%")

# Save final test predictions
save_predictions(test_predictions, test_targets, test_probabilities, 'test')

final_test_acc = (total_test_correct / total_test_samples) * 100

# Calculate class-wise accuracy
test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)
class_accuracy = {}
for class_id in np.unique(test_targets):
    class_mask = test_targets == class_id
    if class_mask.sum() > 0:
        class_accuracy[class_id] = (test_predictions[class_mask] == test_targets[class_mask]).mean() * 100

# Final results logging
logger.info("--- FINAL TEST RESULTS ---")
logger.info(f"Test Accuracy: {final_test_acc:.2f}%")
logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# Save final results
final_results = {
    'final_test_accuracy': final_test_acc,
    'best_validation_accuracy': best_val_acc,
    'class_wise_accuracy': class_accuracy,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'final_epoch': len(epoch_data['epochs'])
}

with open(f'logs/{experiment_name}_final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

# Create final comprehensive plot
create_training_plots(epoch_data, experiment_name)

# Log class-wise accuracy sample
logger.info("Class-wise Accuracy (Sample):")
for i, (class_id, acc) in enumerate(list(class_accuracy.items())[:10]):
    logger.info(f"Class {class_id}: {acc:.2f}%")

# Close TensorBoard writer
writer.close()

logger.info("--- All Done ---")
logger.info(f"Logs saved in: logs/{experiment_name}_*")
logger.info(f"Plots saved in: plots/{experiment_name}_*")
logger.info(f"Predictions saved in: predictions/{experiment_name}_*")
logger.info(f"TensorBoard logs: runs/{experiment_name}")