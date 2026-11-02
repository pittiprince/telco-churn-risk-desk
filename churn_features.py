"""Feature preparation shared by training (this notebook) and serving (app.py).

prepare_features(df)
    takes  : a pandas DataFrame of raw customer records, one row per customer, with the
             columns of Telco_customer_churn.xlsx (names may use spaces or underscores;
             extra columns such as CustomerID or Churn Value are allowed and ignored)
    returns: a new DataFrame with exactly the 22 model input columns in MODEL_COLUMNS,
             same index as the input
    raises : ValueError if one of the 19 required raw columns is missing
"""
import pandas as pd

SERVICE_COLS = ["Phone_Service", "Multiple_Lines", "Online_Security", "Online_Backup",
                "Device_Protection", "Tech_Support", "Streaming_TV", "Streaming_Movies"]

CATEGORICAL_FEATURES = ["Gender", "Senior_Citizen", "Partner", "Dependents", "Phone_Service",
                        "Multiple_Lines", "Internet_Service", "Online_Security", "Online_Backup",
                        "Device_Protection", "Tech_Support", "Streaming_TV", "Streaming_Movies",
                        "Contract", "Paperless_Billing", "Payment_Method"]
RAW_NUMERIC = ["Tenure_Months", "Monthly_Charges", "Total_Charges"]
ENGINEERED = ["Num_Services", "Avg_Monthly_Spend", "Has_Protection"]

# the 19 columns a caller must supply
RAW_INPUT_COLUMNS = (CATEGORICAL_FEATURES[:4] + ["Tenure_Months"] + CATEGORICAL_FEATURES[4:]
                     + ["Monthly_Charges", "Total_Charges"])
# the 22 columns the fitted preprocessor expects, in the same order as X in Phase 1
MODEL_COLUMNS = RAW_INPUT_COLUMNS + ENGINEERED


def standardise_columns(df):
    """'Tenure Months' -> 'Tenure_Months' (same rule as Phase 1, step 1)."""
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out


def prepare_features(df):
    d = standardise_columns(df)
    missing = [c for c in RAW_INPUT_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # numbers can arrive as text (CSV upload, web form) -> coerce
    for c in RAW_NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Phase 1 step 2: blank Total_Charges only happens for brand-new (tenure 0) customers -> 0
    d.loc[(d["Tenure_Months"] == 0) & d["Total_Charges"].isna(), "Total_Charges"] = 0.0

    # Phase 1 step 4: row-wise engineered features (no statistics learned from other rows)
    d["Num_Services"] = (d[SERVICE_COLS] == "Yes").sum(axis=1) + (d["Internet_Service"] != "No").astype(int)
    d["Avg_Monthly_Spend"] = d["Total_Charges"] / d["Tenure_Months"].clip(lower=1)
    d["Has_Protection"] = ((d["Online_Security"] == "Yes") | (d["Tech_Support"] == "Yes")).astype(int)
    return d[MODEL_COLUMNS]
