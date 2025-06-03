from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(X, y, test_size=0.3, random_state=42):
    """
    Scales features and splits data into train/test.
    Returns: X_train, X_test, y_train, y_test
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
