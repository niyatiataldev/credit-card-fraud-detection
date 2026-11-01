
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "random_forest_fraud_model.joblib"
)

model = joblib.load(MODEL_PATH)

# This is a demonstration using a row from the original dataset.
# It is not a manually entered real-world transaction.

df = pd.read_csv(PROJECT_DIR / "data" / "creditcard.csv")

sample = df.drop(columns=["Class"]).iloc[[0]]

prediction = model.predict(sample)[0]
fraud_probability = model.predict_proba(sample)[0, 1]

print("Predicted class:", prediction)
print("Fraud probability:", round(fraud_probability, 4))

if prediction == 1:
    print("Model result: Potential Fraud")
else:
    print("Model result: Normal Transaction")
