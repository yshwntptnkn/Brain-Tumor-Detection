import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_gen):
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=list(test_gen.class_indices.keys())
    )

    return cm, report
