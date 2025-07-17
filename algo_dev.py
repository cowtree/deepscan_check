# Iris Classification Pipeline with Hyperparameter Optimization
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# 3. Create base pipelines for optimization
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 4. Define hyperparameter grids
lr_param_grid = {
    'model__C': [0.1, 1.0, 10.0, 100.0],
    'model__solver': ['liblinear', 'lbfgs'],
    'model__max_iter': [100, 200, 300],
    'model__multi_class': ['ovr', 'multinomial']
}

rf_param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

# 5. Perform hyperparameter optimization
print("\nOptimizing Logistic Regression hyperparameters...")
lr_grid_search = GridSearchCV(
    lr_pipeline, 
    lr_param_grid, 
    cv=5, 
    scoring='accuracy', 
    verbose=1,
    n_jobs=-1
)
lr_grid_search.fit(X_train, y_train)

print("\nBest Logistic Regression parameters:")
print(lr_grid_search.best_params_)
print(f"Best CV accuracy: {lr_grid_search.best_score_:.4f}")

print("\nOptimizing Random Forest hyperparameters...")
rf_grid_search = GridSearchCV(
    rf_pipeline, 
    rf_param_grid, 
    cv=5, 
    scoring='accuracy', 
    verbose=1,
    n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(rf_grid_search.best_params_)
print(f"Best CV accuracy: {rf_grid_search.best_score_:.4f}")

# 6. Select the better model based on CV score
if rf_grid_search.best_score_ > lr_grid_search.best_score_:
    print("\nOptimized Random Forest performed better in cross-validation. Selecting this model.")
    best_model = rf_grid_search.best_estimator_
    best_model_name = "Random Forest"
    best_params = rf_grid_search.best_params_
    best_cv_score = rf_grid_search.best_score_
else:
    print("\nOptimized Logistic Regression performed better in cross-validation. Selecting this model.")
    best_model = lr_grid_search.best_estimator_
    best_model_name = "Logistic Regression"
    best_params = lr_grid_search.best_params_
    best_cv_score = lr_grid_search.best_score_

# 7. Evaluate the selected model on test data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOptimized {best_model_name} performance on test data:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 8. Extract feature importance
if best_model_name == "Logistic Regression":
    coefficients = best_model.named_steps['model'].coef_
    feature_importance = []
    for i, target in enumerate(target_names):
        for j, feature in enumerate(feature_names):
            feature_importance.append({
                'class': target,
                'feature': feature,
                'importance': abs(coefficients[i, j])
            })
    
    importance_df = pd.DataFrame(feature_importance)
    print("\nFeature Coefficients (absolute values for importance):")
    print(importance_df.sort_values('importance', ascending=False))
    
    # Global feature importance (sum of absolute coefficients across classes)
    global_importance = importance_df.groupby('feature')['importance'].sum().reset_index()
    global_importance = global_importance.sort_values('importance', ascending=False)
    
else:  # Random Forest
    importances = best_model.named_steps['model'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    global_importance = feature_importance

# 9. Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=global_importance)
plt.title(f'Feature Importance - Optimized {best_model_name}')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as feature_importance.png")

# 10. Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

# 11. Save model
model_version = "1.0.0"
model_filename = f"iris_classifier_optimized_{best_model_name.replace(' ', '_').lower()}_v{model_version}.joblib"
joblib.dump(best_model, model_filename)
print(f"\nOptimized model saved as {model_filename}")

# 12. Export model metadata for model card
model_metadata = {
    "name": "Iris Flower Classifier",
    "version": model_version,
    "date_created": datetime.now().strftime("%Y-%m-%d"),
    "model_type": "Classification",
    "algorithm": f"Optimized {best_model_name}",
    "features": list(feature_names),
    "target_classes": list(target_names),
    "hyperparameters": {k.replace('model__', ''): v for k, v in best_params.items()},
    "cv_accuracy": float(best_cv_score),
    "performance": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "feature_importance": global_importance.to_dict('records'),
    "dataset_size": len(X),
    "training_size": len(X_train),
    "test_size": len(X_test)
}

# Export metadata to JSON
import json
with open("model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)
print("Model metadata exported to model_metadata.json")