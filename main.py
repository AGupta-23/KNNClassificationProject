from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.model import train_knn, predict
from src.evaluate import evaluate_model
from src.visualize import plot_accuracy_vs_k, plot_decision_boundaries

def main():
    # Step 1: Load Data
    X, y = load_dataset("data/Iris.csv")
    
    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Step 3: Train Model
    model = train_knn(X_train, y_train, k=5)
    
    # Step 4: Predict
    y_pred = predict(model, X_test)
    
    # Step 5: Evaluate
    evaluate_model(y_test, y_pred)
    
    # Step 6: Visualize
    plot_accuracy_vs_k(X_train, X_test, y_train, y_test)
    plot_decision_boundaries(X.values, y.values, k=5)

if __name__ == "__main__":
    main()
