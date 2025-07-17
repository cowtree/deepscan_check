# Iris Classification Pipeline using Logistic Regression
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# 1. Load and prepare data
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 3. Create and train model pipeline
model_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200, random_state=42, multi_class='multinomial', solver='lbfgs'))
])

print("Training model...")
model_pipeline.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Model performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Feature importance (coefficients for logistic regression)
coefficients = model_pipeline.named_steps['model'].coef_
feature_importance = []
for i, target in enumerate(target_names):
    for j, feature in enumerate(feature_names):
        feature_importance.append({
            'class': target,
            'feature': feature,
            'coefficient': coefficients[i, j]
        })

importance_df = pd.DataFrame(feature_importance)
print("\nFeature Coefficients:")
print(importance_df)

# 6. Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

# 7. Save model
model_version = "1.0.0"
model_filename = f"iris_classifier_v{model_version}.joblib"
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved as {model_filename}")

# 8. Export model metadata for model card
model_metadata = {
    "name": "Iris Flower Classifier",
    "version": model_version,
    "date_created": datetime.now().strftime("%Y-%m-%d"),
    "model_type": "Classification",
    "algorithm": "Logistic Regression",
    "features": feature_names,
    "target_classes": target_names.tolist(),
    "performance": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "dataset_size": len(X),
    "training_size": len(X_train),
    "test_size": len(X_test)
}

# Export metadata to JSON
import json
with open("model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)
print("Model metadata exported to model_metadata.json")