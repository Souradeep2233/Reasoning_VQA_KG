# Lightweight KRISP Reproduction

A minimal reproduction of Facebook AI's KRISP (Knowledge-Enhanced Vision-Language Models) with ~75% performance using only 22% of parameters. Built for resource-constrained environments and edge deployment.

## Overview

This project re-examines KRISP's knowledge-grounded VQA approach through aggressive parameter reduction. Instead of industrial-scale training, we demonstrate that structured knowledge integration remains effective even in lightweight architectures suitable for smartphones and AR/VR devices.

**Key Results:**
- Model A: 75% of original KRISP performance with 25M parameters (vs 116M)
- Model B: 3.6M parameters for edge deployment scenarios
- Validated on VQAv2, DAQUAR, and synthetic VQA datasets

## Architecture

Both models use frozen CLIP encoders with trainable projection layers, knowledge retrieval from ConceptNet, and multi-head attention fusion. The core difference:

**Model A:** Direct knowledge concatenation with cross-attention  
**Model B:** Two-stage attention (visual-question fusion → knowledge grounding)

![Architecture Comparison](Our%20Models/Images/architecture.png)

![Model Performance](Our%20Models/Images/performance.png)

## Repository Structure

```
KRISP_REPRODUCTION/          # Core MMF-based framework
├── mmf/                     # Datasets, models, trainers
├── projects/krisp/          # KRISP implementation
└── inf.py                   # Inference script

Our Models/
├── MODEL A/                 # Best performing variant (75% accuracy)
│   ├── arch_KG.py          # Knowledge-grounded architecture
│   ├── train_vqa.py        # VQAv2 training
│   └── run_eval.py         # Evaluation pipeline
└── MODEL B/                 # Lightweight variant (3.6M params)
    └── ee782-model-b-c.ipynb

Predictions/                 # Prediction analysis tools
```

## Results

| Model | Parameters | Dataset | Accuracy |
|-------|-----------|---------|----------|
| KRISP (baseline) | 116M | OKVQA | 32.37% |
| Model A | 25M | VQAv2 | 74.14%* |
| Model B | 3.6M | DAQUAR | 27.75%* |

*Relative to original KRISP baseline

![Training Dynamics](Our%20Models/Images/training.png)

![Sample Outputs](Our%20Models/Images/predictions.png)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train Model A
cd "Our Models/MODEL A"
python train_vqa.py

# Evaluate
python run_eval.py
```

## Key Findings

1. **Parameter Efficiency:** Core KRISP reasoning survives 80% parameter reduction
2. **Image-Grounded Retrieval:** Conditioning knowledge on visual concepts reduces hallucinations
3. **Two-Stage Fusion:** Prevents knowledge from overwhelming visual features
4. **Scale Threshold:** Extreme compression (<5M params) hits fundamental capacity limits

## Technical Details

**Knowledge Integration:**
- CLIP zero-shot detection identifies image concepts
- ConceptNet retrieval anchored to detected objects + question entities
- Top-5 relevant triples per query to minimize noise

**Training:**
- Frozen CLIP ViT-B/32 (visual) + CLIP text encoder
- Knowledge embeddings: 300D (vs 768D in original)
- 8-head cross-attention for fusion
- BERT-based answer decoder
