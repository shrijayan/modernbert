# model_utils.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import numpy as np
import torch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_multilabel_metrics(pred):
    labels = pred.label_ids
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(torch.Tensor(pred.predictions))
    preds = (probs > 0.5).int().numpy()
    
    # Split predictions and labels for severity and action
    severity_labels = labels[:, :4]  # First 4 elements are one-hot severity
    action_labels = labels[:, 4]     # Last element is action
    severity_preds = preds[:, :4]
    action_preds = preds[:, 4]
    
    # Convert one-hot severity predictions to class indices
    severity_label_idx = np.argmax(severity_labels, axis=1)
    severity_pred_idx = np.argmax(severity_preds, axis=1)
    
    # Calculate metrics
    severity_f1 = f1_score(severity_label_idx, severity_pred_idx, average='weighted')
    action_f1 = f1_score(action_labels, action_preds, average='binary')
    
    return {
        'severity_f1': severity_f1,
        'action_f1': action_f1,
        'avg_f1': (severity_f1 + action_f1) / 2
    }
