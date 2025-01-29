# Financial News Sentiment Analysis Model Pipeline 🚀
A knowledge distillation pipeline for creating a lightweight Portuguese financial sentiment analysis model, transferring knowledge from DeepSeek V3 to BERT base.

## 📋 Overview
This project implements an end-to-end pipeline for:
1. Annotating Brazilian Portuguese financial news using DeepSeek V3
2. Training a smaller BERT model through knowledge distillation
3. Creating a sentiment analysis small language model

## 🎯 Performance
Our model achieves remarkable results through knowledge distillation:

- **DeepSeek V3 (Teacher)**: 78.67% accuracy
- **Fine-tuned BERT (Student)**: 78.00% accuracy

**Key Achievement**: We successfully compressed the model from 685B parameters to 110M parameters (6227x reduction) while maintaining nearly identical performance (only 0.67% accuracy drop).

## 🔍 Model Details
- **Teacher Model**: DeepSeek V3 (685B parameters)
- **Student Model**: BERT base (neuralmind/bert-base-portuguese-cased, 110M parameters)
- **Task**: Financial News Sentiment Analysis
- **Languages**: Brazilian Portuguese
- **Model Card**: [winderfeld/bert-portuguese-deepseek-sentiment-analysis](https://huggingface.co/winderfeld/bert-portuguese-deepseek-sentiment-analysis)

## 📊 Dataset

- **Source**: Financial Phrase Bank (Portuguese translation)
- **Classes**: Positive, Negative, Neutral
- **Training samples**: 3000
- **Test samples**: 300 (100 per class)
- **Data Format**: CSV with columns:
  - text: Original financial news text
  - label_text: Original sentiment label
  - deepseek_reason: Model's reasoning
  - deepseek_label_text: Predicted sentiment
    
## 🛠️ Training Pipeline

### 1. Data Annotation
- **Environment**: Google Colab
- **Annotation Model**: DeepSeek V3
- **Features**:
  - Robust error handling
  - Automatic retries
  - Progress tracking
  - Rate limiting
  - Structured output

### 2. Model Training
- **Platform**: HuggingFace AutoTrain Advanced
- **Infrastructure**: Google Colab GPU
- **Process**:
  - Knowledge distillation from DeepSeek V3
  - Fine-tuning on financial domain
  - Optimized for Portuguese language

### 3. Training Configuration
```python
{
    "auto_find_batch_size": "false",
    "eval_strategy": "epoch",
    "mixed_precision": "fp16",
    "optimizer": "adamw_torch",
    "scheduler": "linear",
    "batch_size": "16",
    "early_stopping_patience": "5",
    "early_stopping_threshold": "0.01",
    "epochs": "5",
    "gradient_accumulation": "1",
    "lr": "0.00005",
    "logging_steps": "-1",
    "max_grad_norm": "1",
    "max_seq_length": "128",
    "save_total_limit": "1",
    "seed": "42",
    "warmup_ratio": "0.1",
    "weight_decay": "0"
}
```

