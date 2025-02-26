# model_utils.py
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def compute_multilabel_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    
    # Split predictions and labels
    severity_labels = labels[:, :4]
    action_labels = labels[:, 4]
    
    # Get severity predictions
    severity_logits = logits[:, :4]
    severity_preds = np.argmax(severity_logits, axis=1)
    severity_true = np.argmax(severity_labels, axis=1)
    
    # Get action predictions
    action_logits = logits[:, 4]
    action_preds = (action_logits > 0).astype(int)
    
    # Calculate metrics for severity
    severity_precision, severity_recall, severity_f1, _ = precision_recall_fscore_support(
        severity_true, severity_preds, average='weighted'
    )
    severity_accuracy = accuracy_score(severity_true, severity_preds)
    
    # Calculate metrics for action
    action_precision, action_recall, action_f1, _ = precision_recall_fscore_support(
        action_labels, action_preds, average='binary'
    )
    action_accuracy = accuracy_score(action_labels, action_preds)
    
    return {
        'severity_accuracy': float(severity_accuracy),
        'severity_f1': float(severity_f1),
        'severity_precision': float(severity_precision),
        'severity_recall': float(severity_recall),
        'action_accuracy': float(action_accuracy),
        'action_f1': float(action_f1),
        'action_precision': float(action_precision),
        'action_recall': float(action_recall),
        'avg_f1': float((severity_f1 + action_f1) / 2)
    }
