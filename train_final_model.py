
from pathlib import Path

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest_fraud_model.joblib"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and clean data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)

X = df.drop(columns=["Class"])
y = df["Class"]

print("\nClass distribution after cleaning:")
print(y.value_counts())


# --------------------------------------------------
# 3. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 4. Apply SMOTE only to training data
# --------------------------------------------------

smote = SMOTE(random_state=RANDOM_STATE)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train,
)

print("\nTraining distribution after SMOTE:")
print(y_train_smote.value_counts())


# --------------------------------------------------
# 5. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train_smote, y_train_smote)

print("\nRandom Forest training completed.")


# --------------------------------------------------
# 6. Evaluate on the original test data
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        digits=4,
    )
)

print("Fraud Precision:", precision_score(y_test, y_pred))
print("Fraud Recall:", recall_score(y_test, y_pred))
print("Fraud F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Average Precision:", average_precision_score(y_test, y_prob))


# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
