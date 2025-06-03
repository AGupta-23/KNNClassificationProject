# 🧠 K-Nearest Neighbors (KNN) Classification Project

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm using the **Iris dataset**. It involves loading data, preprocessing, training a classifier, evaluating performance, and visualizing results.

---

## 📁 Project Structure

KNNClassificationProject/
│
├── data/
│   └── Iris.csv                  # Dataset file
│
├── src/
│   ├── load_data.py              # Function to load dataset
│   ├── preprocess.py             # Normalize features
│   ├── model.py                  # KNN model training
│   ├── evaluate.py               # Accuracy, confusion matrix, classification report
│   └── visualize.py              # Accuracy vs K, decision boundaries
│
├── main.py                       # Main entry script
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

## 📦 Requirements

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
📊 Dataset
Name: Iris Dataset

Source: UCI Machine Learning Repository

Features: Sepal length, Sepal width, Petal length, Petal width

Target: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

🚀 How to Run
Run the project from the root folder:

bash
Copy
Edit
python main.py
You will see:

Console output: Accuracy, Confusion Matrix, Classification Report

Plots: Accuracy vs. K graph, Decision Boundaries (2D)

🔍 Features
✅ Load and preprocess the dataset
✅ Normalize features using MinMaxScaler
✅ Train KNN classifier using KNeighborsClassifier
✅ Test multiple values of K
✅ Plot accuracy vs. K to find the best K
✅ Display confusion matrix and classification report
✅ 2D decision boundary visualization

📈 Sample Output
text
Copy
Edit
Accuracy: 1.0
Confusion Matrix:
 [[19  0  0]
  [ 0 13  0]
  [ 0  0 13]]
Classification Report:
                  precision    recall  f1-score   support
    Iris-setosa       1.00      1.00      1.00        19
Iris-versicolor       1.00      1.00      1.00        13
 Iris-virginica       1.00      1.00      1.00        13
🛠️ Technologies Used
Python 3.x

Scikit-learn

Pandas

Matplotlib

Seaborn

📚 Learnings
Understanding of how KNN works

Importance of feature normalization

Model evaluation metrics

How to visualize model performance and decision boundaries

✨ Future Enhancements
Add CLI to allow custom K input

Cross-validation for better evaluation

Apply KNN to a more complex dataset

Export trained model using joblib

🧑‍💻 Author
ABHIDHA GUPTA
CSE Student | AI & ML Enthusiast