# Churn Risk Desk

Streamlit app for the Telco customer churn mini project (MCA 5141 ML Lab, MIT Manipal) by Pitti Prince Charles.

A tuned XGBoost pipeline predicts the probability that a telecom customer leaves next month
(test ROC-AUC 0.856; recall 0.757 and precision 0.544 at the 0.56 cut-off), trained on the
IBM Telco Customer Churn dataset (7,043 customers).

- **Customer**: score one customer and see which retention offer lowers the risk most
- **File**: score a CSV or Excel file of customers (30 demo customers included)
- **Model**: metrics, drivers and limits

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.
