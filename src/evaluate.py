from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_test, y_pred):
    """
    Prints accuracy, confusion matrix, and classification report.
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
