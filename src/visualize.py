import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

def plot_accuracy_vs_k(X_train, X_test, y_train, y_test, max_k=20):
    accuracies = []
    k_range = range(1, max_k+1)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)

    plt.figure()
    plt.plot(k_range, accuracies, marker='o')
    plt.title('Accuracy vs K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def plot_decision_boundaries(X, y, k=5, feature_indices=(0, 1)):
    """
    Visualizes decision boundaries using 2 selected features
    """
    X = X[:, feature_indices]  # select only two features for visualization
    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(f"Decision Boundary (K={k})")
    plt.xlabel(f"Feature {feature_indices[0]}")
    plt.ylabel(f"Feature {feature_indices[1]}")
    plt.show()
