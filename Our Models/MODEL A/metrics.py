import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    top_k_accuracy_score
)
import warnings

# Suppress UndefinedMetricWarning for classes with no predictions
warnings.filterwarnings("ignore")

class VQAMetricsEvaluator:
    def __init__(self, model, device, tokenizer=None):
        """
        Args:
            model: The trained PyTorch model.
            device: 'cuda' or 'cpu'.
            tokenizer: Optional BERT tokenizer (to decode class labels if they are tokens).
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.results = {}

    def get_predictions(self, dataloader):
        """
        Runs inference on the dataloader and collects logits and labels.
        """
        self.model.eval()
        all_logits = []
        all_preds = []
        all_labels = []
        
        print("--- Running Inference for Metrics ---")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Unpack based on your specific dataloader structure
                images, questions, cn_vectors, labels = batch
                
                images = images.to(self.device)
                questions = questions.to(self.device)
                cn_vectors = cn_vectors.to(self.device).float()
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(images, questions, cn_vectors)
                
                # Store data
                all_logits.append(logits.cpu())
                all_preds.append(torch.argmax(logits, dim=1).cpu())
                all_labels.append(labels.cpu())

        # Concatenate all batches
        self.y_logits = torch.cat(all_logits).numpy()
        self.y_pred = torch.cat(all_preds).numpy()
        self.y_true = torch.cat(all_labels).numpy()
        
        # Calculate probabilities for Top-K (using Softmax)
        self.y_probs = torch.nn.functional.softmax(torch.from_numpy(self.y_logits), dim=1).numpy()

    def calculate_metrics(self):
        """
        Calculates comprehensive classification metrics.
        """
        if not hasattr(self, 'y_pred'):
            raise ValueError("Run get_predictions() first.")

        print("\n--- Calculating Metrics ---")
        
        # 1. Standard Accuracy
        acc = accuracy_score(self.y_true, self.y_pred)
        
        # 2. Top-K Accuracy (Top-5 is standard for VQA/ImageNet)
        # Handle case where num_classes < 5
        n_classes = self.y_logits.shape[1]
        k = min(5, n_classes)
        top_k_acc = top_k_accuracy_score(self.y_true, self.y_probs, k=k)

        # 3. Precision, Recall, F1
        # 'macro': calculates metrics for each label, and finds their unweighted mean. 
        # 'weighted': calculates metrics for each label, and finds their average weighted by support.
        prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro')
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='weighted')

        self.results = {
            "Accuracy (Top-1)": acc * 100,
            f"Accuracy (Top-{k})": top_k_acc * 100,
            "Precision (Macro)": prec_mac,
            "Recall (Macro)": rec_mac,
            "F1 Score (Macro)": f1_mac,
            "Precision (Weighted)": prec_w,
            "Recall (Weighted)": rec_w,
            "F1 Score (Weighted)": f1_w
        }

        return self.results

    def print_summary(self):
        print("\n" + "="*40)
        print("       FINAL PERFORMANCE METRICS       ")
        print("="*40)
        for metric, value in self.results.items():
            if "Accuracy" in metric:
                print(f"{metric:<25}: {value:.2f}%")
            else:
                print(f"{metric:<25}: {value:.4f}")
        print("="*40)

    def save_class_report(self, filename="class_performance.csv"):
        """
        Saves a detailed CSV of how the model performed on EVERY individual class.
        """
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename)
        print(f"\nDetailed class report saved to: {filename}")

    def plot_confusion_matrix(self, top_n_classes=20, save_path="confusion_matrix.png"):
        """
        Plots a confusion matrix. 
        If there are too many classes, it only plots the 'top_n_classes' most frequent ones.
        """
        # Get the most frequent classes in the test set to avoid a massive unreadable plot
        class_counts = pd.Series(self.y_true).value_counts()
        top_classes = class_counts.head(top_n_classes).index.tolist()
        
        # Filter data for these classes
        mask = np.isin(self.y_true, top_classes) & np.isin(self.y_pred, top_classes)
        filtered_true = self.y_true[mask]
        filtered_pred = self.y_pred[mask]
        
        cm = confusion_matrix(filtered_true, filtered_pred, labels=top_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=top_classes, yticklabels=top_classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Top {top_n_classes} Frequent Classes)')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Confusion Matrix saved to: {save_path}")

    def analyze_errors(self, k=5):
        """
        Identifies the top confused pairs (e.g., predicted 'dog' but was 'cat').
        """
        mask = self.y_pred != self.y_true
        wrong_preds = self.y_pred[mask]
        wrong_trues = self.y_true[mask]
        
        pairs = list(zip(wrong_trues, wrong_preds))
        from collections import Counter
        pair_counts = Counter(pairs)
        
        print(f"\n--- Top {k} Most Common Errors ---")
        print(f"{'True Label':<15} | {'Predicted':<15} | {'Count'}")
        print("-" * 45)
        for (truth, pred), count in pair_counts.most_common(k):
            print(f"{truth:<15} | {pred:<15} | {count}")