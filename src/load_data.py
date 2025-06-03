import pandas as pd

def load_dataset(filepath):
    """
    Loads the dataset from the given filepath.
    Returns: features (X), labels (y)
    """
    data = pd.read_csv(filepath)
    
    # Assuming the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
