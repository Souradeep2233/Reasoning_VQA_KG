import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import clip
from transformers import BertConfig, BertForMaskedLM
from metrics import VQAMetricsEvaluator

# --- IMPORT YOUR ARCHITECTURE ---
# Assuming your original script is 'main.py' and the class is 'ImprovedModel'
# If you cannot import from main, copy the 'ImprovedModel' class definition here.
from arch_KG_back_prop import ImprovedModel, get_vqa_loader, ConceptNetFeatureExtractor, preprocess, bert_tokenizer

# --- CONFIGURATION ---
CONCEPTNET_EMBEDDING_PATH = "/home/souradeepd/Desktop/New_arch_KIT/numberbatch-en-19.08.txt"
MODEL_PATH = "best_model.pth" # Path to the saved .pth file from training
TEST_SPLIT = "validation[50:100]"
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_for_eval():
    print("--- Re-instantiating Model Architecture ---")
    
    # 1. Re-load Base Components
    clip_model, _ = clip.load("ViT-B/32", device=device)
    bert_model_name = 'bert-base-uncased'
    bert_for_mlm = BertForMaskedLM.from_pretrained(bert_model_name)
    language_head = bert_for_mlm.cls.to(device)
    
    # 2. Re-load ConceptNet
    feature_extractor = ConceptNetFeatureExtractor(CONCEPTNET_EMBEDDING_PATH)
    cn_embed_dim = feature_extractor.embed_dim
    
    # 3. Initialize Base Model
    model = ImprovedModel(clip_model, language_head, cn_embed_dim).to(device)
    
    # 4. Re-apply LoRA (Must match training config exactly)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "image_linear.0", "text_linear.0",
            "fusion_layers.0", "fusion_layers.4",
            "projection_to_bert.0"
        ],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["cross_attention", "language_head"]
    )
    model = get_peft_model(model, lora_config)
    
    # 5. Load Weights
    print(f"Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    
    return model, feature_extractor

def main():
    # 1. Load Model & Data Tools
    model, feature_extractor = load_model_for_eval()
    
    # 2. Get Test Loader
    print("Loading Test Data...")
    test_loader = get_vqa_loader(
        batch_size=BATCH_SIZE,
        clip_preprocess=preprocess,
        bert_tokenizer=bert_tokenizer,
        feature_extractor=feature_extractor,
        split=TEST_SPLIT,
        shuffle=False
    )
    
    # 3. Instantiate Evaluator
    evaluator = VQAMetricsEvaluator(model, device)
    
    # 4. Run Inference
    evaluator.get_predictions(test_loader)
    
    # 5. Calculate & Print Metrics
    evaluator.calculate_metrics()
    evaluator.print_summary()
    
    # 6. Save Detailed Reports
    evaluator.save_class_report("final_test_results_per_class.csv")
    
    # 7. Visualizations
    evaluator.plot_confusion_matrix(top_n_classes=15)
    evaluator.analyze_errors(k=10)

if __name__ == "__main__":
    main()