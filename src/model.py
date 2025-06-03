from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, k=3):
    """
    Trains KNN classifier with specified k.
    Returns: trained model
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def predict(model, X_test):
    """
    Predicts using the trained model.
    """
    return model.predict(X_test)
