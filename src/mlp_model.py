from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def mlp_model(X, y):
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        {"hidden_layer_sizes": (32,), "description": "1 gizli katman, 32 nöron"},
        {"hidden_layer_sizes": (32, 32), "description": "2 gizli katman, 32'şer nöron"},
        {"hidden_layer_sizes": (32, 32, 32), "description": "3 gizli katman, 32'şer nöron"}
    ]

    for config in configs:
        mlp = MLPClassifier(hidden_layer_sizes=config["hidden_layer_sizes"], max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        results.append((config["description"], accuracy, precision, recall, f1))
    
    return results
