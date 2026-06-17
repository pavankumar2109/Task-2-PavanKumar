# ============================================================
# PROJECT 2: Data Classification Using AI
# DecodeLabs Batch 2026 | KNN on Iris Dataset
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# ─────────────────────────────────────────────
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading the Iris Dataset")
print("=" * 50)

iris = load_iris()

# Convert to DataFrame for easy viewing
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]

print(f"\nDataset shape: {df.shape}")
print(f"Classes: {list(iris.target_names)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nClass distribution:")
print(df['species'].value_counts())


# ─────────────────────────────────────────────
# STEP 2: PREPARE FEATURES AND LABELS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Preparing Features (X) and Labels (y)")
print("=" * 50)

X = iris.data    # Features: 4 measurements
y = iris.target  # Labels: 0, 1, or 2 (species index)

print(f"X shape: {X.shape} → {X.shape[0]} samples, {X.shape[1]} features")
print(f"y shape: {y.shape} → target class for each sample")


# ─────────────────────────────────────────────
# STEP 3: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Splitting into Train (80%) and Test (20%)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Fixed seed for reproducibility
    shuffle=True        # Shuffle to remove order bias
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")


# ─────────────────────────────────────────────
# STEP 4: FEATURE SCALING
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Scaling Features with StandardScaler")
print("=" * 50)

scaler = StandardScaler()

# IMPORTANT: Fit on training data only, then transform both
# (Never fit on test data — that's data leakage!)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Before scaling (first row of X_test):", X_test[0].round(2))
print("After scaling: mean ≈ 0, std ≈ 1 across features")


# ─────────────────────────────────────────────
# STEP 5: FIND THE BEST K (Elbow Method)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Finding Optimal K (Elbow Method)")
print("=" * 50)

error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, pred))

# Plot the elbow
plt.figure(figsize=(8, 4))
plt.plot(k_range, error_rates, marker='o', color='navy', linestyle='--')
plt.title('Elbow Method: Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150)
plt.show()
print("Best K is at the 'elbow' — lowest error point.")


# ─────────────────────────────────────────────
# STEP 6: TRAIN THE KNN MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Training the KNN Classifier (k=5)")
print("=" * 50)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)      # FIT: memorize the map
predictions = model.predict(X_test)  # PREDICT: apply logic

print("Model trained successfully!")
print(f"Sample predictions: {predictions[:10]}")
print(f"Actual labels:      {y_test[:10]}")


# ─────────────────────────────────────────────
# STEP 7: EVALUATE THE MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7: Evaluating Model Performance")
print("=" * 50)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"F1 Score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))


# ─────────────────────────────────────────────
# STEP 8: CONFUSION MATRIX (VISUALIZATION)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8: Confusion Matrix")
print("=" * 50)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title('Confusion Matrix — KNN on Iris Dataset')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("Confusion matrix saved as confusion_matrix.png")
print("\nReading the matrix:")
print("  Diagonal = correct predictions (TP)")
print("  Off-diagonal = mistakes (FP/FN)")


# ─────────────────────────────────────────────
# STEP 9: PREDICT A NEW CUSTOM FLOWER
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 9: Predicting a New, Unseen Flower")
print("=" * 50)

# New flower: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)
probability = model.predict_proba(new_flower_scaled)

print(f"New flower measurements: {new_flower[0]}")
print(f"Predicted species: {iris.target_names[prediction[0]]}")
print(f"Confidence: {probability[0].max() * 100:.1f}%")

print("\n" + "=" * 50)
print("PROJECT 2 COMPLETE!")
print("=" * 50)