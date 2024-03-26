import numpy as np
import evaluate

metric=evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions=np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
