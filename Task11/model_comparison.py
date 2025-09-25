import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize for KNN & Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression (Baseline)
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)

# Decision Tree (Alternative)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

# K-Nearest Neighbors (Alternative)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_preds)

# Model Comparison Table
comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression (Baseline)",
        "Decision Tree (Alternative)",
        "K-Nearest Neighbors (Alternative)"
    ],
    "Accuracy (%)": [
        round(lr_acc * 100, 2),
        round(dt_acc * 100, 2),
        round(knn_acc * 100, 2)
    ],
    "Comments": [
        "Standard linear classifier",
        "Tree-based splits; good for non-linearity",
        "Distance-based method; needs scaling"
    ]
})
comparison_df.to_csv("model_comparison.csv", index=False)

# Save predictions
preds_df = pd.DataFrame({
    "Actual": y_test,
    "Logistic Regression": lr_preds,
    "Decision Tree": dt_preds,
    "KNN": knn_preds
})
preds_df.to_csv("predictions.csv", index=False)

# Confusion matrix (for Decision Tree)
cm = confusion_matrix(y_test, dt_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("decision_tree_confusion_matrix.png")
plt.close() 

print("✅ All files generated successfully!")
print("Comparison Table: model_comparison.csv")

