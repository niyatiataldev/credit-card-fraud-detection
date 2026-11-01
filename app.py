
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# --------------------------------------------------
# 1. Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_fraud_model.joblib"


# --------------------------------------------------
# 2. Load dataset and model
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# --------------------------------------------------
# 3. Evaluate model on the same test split
# --------------------------------------------------

@st.cache_data
def get_model_results():
    clean_df = df.drop_duplicates().copy()

    X = clean_df.drop(columns=["Class"])
    y = clean_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Average Precision": average_precision_score(
            y_test, y_prob
        ),
    }

    return metrics, cm, len(X_test), int(y_test.sum())


metrics, cm, test_count, test_fraud_count = get_model_results()


# --------------------------------------------------
# 4. Dataset overview calculations
# --------------------------------------------------

normal = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

total_transactions = len(df)
normal_count = len(normal)
fraud_count = len(fraud)

fraud_percentage = (fraud_count / total_transactions) * 100

average_normal_amount = normal["Amount"].mean()
average_fraud_amount = fraud["Amount"].mean()

median_normal_amount = normal["Amount"].median()
median_fraud_amount = fraud["Amount"].median()


# --------------------------------------------------
# 5. Dashboard title
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection Dashboard")

st.write(
    "Explore the transaction dataset and evaluate how well "
    "a Random Forest model identifies fraudulent transactions."
)

st.info(
    "Fraudulent transactions are rare. A model can achieve high "
    "overall accuracy while still missing some fraud cases."
)


# --------------------------------------------------
# 6. Dataset overview
# --------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}",
)

col2.metric(
    "Normal Transactions",
    f"{normal_count:,}",
)

col3.metric(
    "Fraud Transactions",
    f"{fraud_count:,}",
)

col4.metric(
    "Fraud Rate",
    f"{fraud_percentage:.3f}%",
)

st.caption(
    "These figures describe the original dataset before duplicate "
    "removal. The model was trained and evaluated after duplicates "
    "were removed."
)


# --------------------------------------------------
# 7. Class distribution
# --------------------------------------------------

st.header("2. Normal vs Fraud Transactions")

st.write(
    "The dataset contains many more normal transactions than "
    "fraudulent transactions."
)

class_data = pd.DataFrame(
    {
        "Transaction Type": ["Normal", "Fraud"],
        "Number of Transactions": [
            normal_count,
            fraud_count,
        ],
    }
)

st.bar_chart(
    class_data,
    x="Transaction Type",
    y="Number of Transactions",
    use_container_width=True,
)

st.warning(
    f"Only {fraud_percentage:.3f}% of transactions are fraudulent. "
    "This is called class imbalance."
)

st.write(
    "Because fraud is rare, we use techniques such as SMOTE "
    "during training to help the model learn from fraudulent cases. "
    "The test data remains in its original distribution."
)


# --------------------------------------------------
# 8. Model performance
# --------------------------------------------------

st.header("3. Random Forest Model Performance")

st.write(
    "The following results measure the model's performance "
    "on the test dataset."
)

st.caption(
    f"The test set contains {test_count:,} transactions, "
    f"including {test_fraud_count} fraudulent transactions."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Fraud Precision",
    f"{metrics['Precision']:.2%}",
)

col2.metric(
    "Fraud Recall",
    f"{metrics['Recall']:.2%}",
)

col3.metric(
    "Fraud F1-score",
    f"{metrics['F1-score']:.2%}",
)

col4, col5 = st.columns(2)

col4.metric(
    "ROC-AUC",
    f"{metrics['ROC-AUC']:.4f}",
)

col5.metric(
    "Average Precision",
    f"{metrics['Average Precision']:.4f}",
)

st.subheader("Confusion Matrix")

confusion_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Fraud"],
    columns=["Predicted Normal", "Predicted Fraud"],
)

st.dataframe(
    confusion_df,
    use_container_width=True,
)

tn, fp, fn, tp = cm.ravel()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Fraud Detected",
    f"{tp}",
)

col2.metric(
    "Fraud Missed",
    f"{fn}",
)

col3.metric(
    "Normal Transactions Flagged",
    f"{fp}",
)

st.write("### Understanding the results")

st.write(
    f"- The model correctly detected **{tp}** fraudulent transactions.\n"
    f"- It missed **{fn}** fraudulent transactions.\n"
    f"- It incorrectly flagged **{fp}** normal transactions as fraud."
)

st.caption(
    "Precision measures how many fraud predictions were correct. "
    "Recall measures how many actual fraud cases were detected. "
    "F1-score combines precision and recall."
)


# --------------------------------------------------
# 9. Transaction amount analysis
# --------------------------------------------------

st.header("4. Transaction Amount Analysis")

st.write(
    "We compare the average and median transaction amounts "
    "for normal and fraudulent transactions."
)

amount_data = pd.DataFrame(
    {
        "Transaction Type": ["Normal", "Fraud"],
        "Average Amount": [
            average_normal_amount,
            average_fraud_amount,
        ],
        "Median Amount": [
            median_normal_amount,
            median_fraud_amount,
        ],
    }
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Transaction Amount")

    st.bar_chart(
        amount_data,
        x="Transaction Type",
        y="Average Amount",
        use_container_width=True,
    )

with col2:
    st.subheader("Median Transaction Amount")

    st.bar_chart(
        amount_data,
        x="Transaction Type",
        y="Median Amount",
        use_container_width=True,
    )

st.caption(
    "Average = total amount divided by transaction count. "
    "Median = the middle transaction amount when amounts are sorted. "
    "The median is less affected by unusually large transactions."
)


# --------------------------------------------------
# 10. Key observations
# --------------------------------------------------

st.header("5. Key Observations")

st.write(
    f"- Normal transactions have an average amount of "
    f"{average_normal_amount:,.2f}.\n"
    f"- Fraudulent transactions have an average amount of "
    f"{average_fraud_amount:,.2f}.\n"
    f"- The median normal transaction amount is "
    f"{median_normal_amount:,.2f}.\n"
    f"- The median fraud transaction amount is "
    f"{median_fraud_amount:,.2f}."
)

st.caption(
    "The Amount column is shown in the dataset's recorded units. "
    "Its currency has not been confirmed."
)
