"""Churn Risk Desk: Streamlit app for the Telco churn mini project (Phase 3, deployment).

Run:  streamlit run app.py
Needs in the same folder: churn_model.joblib, churn_features.py  (both created by the notebook)
"""
import html
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import xgboost

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # joblib must be able to import churn_features
from churn_features import RAW_INPUT_COLUMNS, standardise_columns  # noqa: E402

MODEL_PATH = os.path.join(HERE, "churn_model.joblib")
SAMPLE_PATH = os.path.join(HERE, "sample_customers.csv")

st.set_page_config(page_title="Churn Risk Desk", layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- look: white minimalism
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500&family=Geist+Mono:wght@400&display=swap">
<style>
:root { --ink:#0d0d0d; --grey:#8a8a8a; --faint:#ececec; --risk:#d5361f; }
html, body, [class*="st-"], .stMarkdown, button, input, label, p, li, td, th { font-family: "Geist", "Helvetica Neue", Arial, sans-serif !important; }
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 1080px; }
h1 { font-weight: 500 !important; font-size: 1.45rem !important; letter-spacing: -.01em; padding: 0 !important; }
h3 { font-weight: 400 !important; font-size: .78rem !important; color: var(--grey) !important; letter-spacing: 0; margin-top: 1.4rem !important; padding: 0 0 .3rem !important; border-bottom: 1px solid var(--faint); }
.sub { color: var(--grey); margin: -.6rem 0 1.6rem; }
/* tabs: plain words with an underline */
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--faint); }
[data-baseweb="tab"] { padding: 0 0 .5rem !important; background: none !important; color: var(--grey) !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--ink) !important; }
[data-baseweb="tab-highlight"] { background: var(--ink) !important; height: 1px !important; }
[data-baseweb="tab-border"] { display: none; }
/* radios: text options, selected one underlined, no circles */
[role="radiogroup"] { gap: .2rem 1.2rem !important; }
[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
[data-testid="stRadioOption"] { padding: 0 0 2px !important; margin: 0 !important; border-bottom: 1px solid transparent; }
[data-testid="stRadioOption"] p { color: var(--grey) !important; font-size: .9rem !important; }
[data-testid="stRadioOption"]:hover p { color: var(--ink) !important; }
[data-testid="stRadioOption"][data-selected="true"] { border-bottom-color: var(--ink); }
[data-testid="stRadioOption"][data-selected="true"] p { color: var(--ink) !important; }
[data-testid="stWidgetLabel"] p { color: var(--grey) !important; font-size: .82rem !important; }
.stRadio { padding: .1rem 0 .5rem; border-bottom: 1px solid var(--faint); }
/* number inputs: underline only */
[data-testid="stNumberInputContainer"] { border: 0 !important; border-bottom: 1px solid var(--faint) !important; border-radius: 0 !important; background: transparent !important; }
[data-testid="stNumberInputContainer"]:focus-within { border-bottom-color: var(--ink) !important; }
[data-testid="stNumberInputField"] { background: transparent !important; font-family: "Geist Mono", monospace !important; padding-left: 0 !important; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { display: none !important; }
/* keep icon glyphs in their own font */
[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
/* sliders: hairline track, small black thumb */
[data-testid="stSlider"] [role="slider"] { background: var(--ink) !important; box-shadow: none !important; width: 11px !important; height: 11px !important; }
[data-testid="stSliderThumbValue"] { color: var(--ink) !important; font-family: "Geist Mono", monospace !important; font-size: .78rem !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div { height: 1px !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--grey) !important; font-size: .72rem !important; }
/* buttons */
.stButton button, .stDownloadButton button { border-radius: 0 !important; border: 1px solid var(--ink) !important; background: #fff !important; color: var(--ink) !important; box-shadow: none !important; }
.stButton button:hover, .stDownloadButton button:hover { background: #f6f6f6 !important; }
[data-testid="stFileUploaderDropzone"] { border-radius: 0 !important; background: #fff !important; border: 1px dashed #cfcfcf !important; }
/* readout */
.lbl { font-size: .78rem; color: var(--grey); }
.big { font-size: 6rem; font-weight: 300; line-height: .95; letter-spacing: -.045em; font-variant-numeric: tabular-nums; }
.big small { font-size: 2.5rem; letter-spacing: -.02em; }
.hot { color: var(--risk) !important; }
.line { position: relative; height: 2px; background: var(--faint); margin: 1.1rem 0 .5rem; }
.line i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--ink); }
.line.hot i { background: var(--risk); }
.line b { position: absolute; top: -5px; width: 1px; height: 12px; background: var(--ink); }
.scale { display: flex; justify-content: space-between; font-family: "Geist Mono", monospace; font-size: .7rem; color: var(--grey); }
.verdict { font-size: 1.15rem; font-weight: 500; margin-top: 1.6rem; }
.note { color: var(--grey); font-size: .88rem; margin: .2rem 0 1.4rem; }
table.min { width: 100%; border-collapse: collapse; font-size: .9rem; border: 0 !important; margin: 0 !important; }
table.min tr { background: transparent !important; border: 0 !important; }
table.min td, table.min th { padding: .5rem 0 !important; border: 0 !important; border-top: 1px solid var(--faint) !important; text-align: left; background: transparent !important; }
table.min td + td, table.min th + th { padding-left: 1rem !important; }
table.min th { border-top: 0 !important; font-weight: 400; color: var(--grey); font-size: .78rem; }
table.min .n { text-align: right !important; font-family: "Geist Mono", monospace; font-size: .82rem; font-variant-numeric: tabular-nums; padding-left: 1rem; white-space: nowrap; }
table.min .g { color: var(--grey); }
.figs { display: flex; gap: 3.5rem; flex-wrap: wrap; margin: 1rem 0; }
.figs b { display: block; font-size: 2.6rem; font-weight: 300; letter-spacing: -.03em; line-height: 1; }
.figs span { font-size: .78rem; color: var(--grey); }
.bar { display: grid; grid-template-columns: 9rem 1fr 3rem; gap: .9rem; align-items: center; padding: .42rem 0; border-top: 1px solid var(--faint); font-size: .9rem; }
.bar i { display: block; height: 1px; background: var(--ink); }
.bar span:last-child { text-align: right; font-family: "Geist Mono", monospace; font-size: .78rem; color: var(--grey); }
.foot { color: var(--grey); font-size: .75rem; border-top: 1px solid var(--faint); padding-top: .8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- loading
@st.cache_resource
def load_bundle(path):
    """takes: path to churn_model.joblib  ->  returns: the saved dict (model, threshold, metadata)."""
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error("churn_model.joblib not found. Run the notebook (Phase 3.3) first, then put the file next to app.py.")
    st.stop()

bundle = load_bundle(MODEL_PATH)
model = bundle["model"]
opts = bundle["category_options"]
TRAIN_RATE = bundle["train_churn_rate"]

saved_v = bundle["versions"]
if saved_v["scikit-learn"] != sklearn.__version__ or saved_v["xgboost"] != xgboost.__version__:
    st.warning(
        f"Model was saved with scikit-learn {saved_v['scikit-learn']} / xgboost {saved_v['xgboost']}, "
        f"but this machine has {sklearn.__version__} / {xgboost.__version__}. "
        "Install the versions in requirements.txt (or re-run the notebook here) to be sure predictions match."
    )


def predict(df):
    """takes: DataFrame of raw customer rows  ->  returns: 1-D numpy array of churn probabilities."""
    return model.predict_proba(df)[:, 1]


def pct(v):
    return f"{v * 100:.1f}%"


if "cut" not in st.session_state:
    st.session_state.cut = float(bundle["threshold"])

st.title("Churn Risk Desk")
st.markdown('<p class="sub">Probability that a telecom customer leaves next month.</p>', unsafe_allow_html=True)

tab_one, tab_batch, tab_card = st.tabs(["Customer", "File", "Model"])

# ============================================================== TAB 1: one customer
with tab_one:
    left, gap, right = st.columns([1.25, 0.12, 1])

    def choice(label, field, default, options=None):
        """takes: label, dataset column, default value  ->  returns: the option the user picked (str)."""
        options = options or opts[field]
        return st.radio(label, options, index=options.index(default), horizontal=True, key=field)

    with left:
        st.markdown("### Customer")
        gender = choice("Gender", "Gender", "Female")
        senior = choice("Senior citizen", "Senior_Citizen", "No")
        partner = choice("Partner", "Partner", "No")
        dependents = choice("Dependents", "Dependents", "No")
        st.markdown("### Account")
        contract = choice("Contract", "Contract", "Month-to-month")
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        st.markdown("### Billing")
        paperless = choice("Paperless billing", "Paperless_Billing", "Yes")
        payment = choice("Payment method", "Payment_Method", "Electronic check")
        c1, c2 = st.columns(2)
        monthly = c1.number_input("Monthly charges ($)", min_value=0.0, max_value=300.0, value=85.0, step=0.05)
        auto_total = c2.checkbox("Total = tenure × monthly", value=True)
        total = round(tenure * monthly, 2) if auto_total else c2.number_input(
            "Total charges ($)", min_value=0.0, value=round(tenure * monthly, 2), step=1.0)
        lo, hi = bundle["numeric_ranges"]["Monthly_Charges"]
        if not lo <= monthly <= hi:
            st.caption(f":red[Outside the training range (${lo:.2f}–${hi:.2f}).]")
        st.markdown("### Services")
        phone = choice("Phone", "Phone_Service", "Yes")
        multi = "No phone service" if phone == "No" else choice(
            "Multiple lines", "Multiple_Lines", "No", [o for o in opts["Multiple_Lines"] if o != "No phone service"])
        internet = choice("Internet", "Internet_Service", "Fiber optic")
        addons = {}
        for a, lab in [("Online_Security", "Online security"), ("Online_Backup", "Online backup"),
                       ("Device_Protection", "Device protection"), ("Tech_Support", "Tech support"),
                       ("Streaming_TV", "Streaming TV"), ("Streaming_Movies", "Streaming movies")]:
            addons[a] = "No internet service" if internet == "No" else choice(
                lab, a, "No", [o for o in opts[a] if o != "No internet service"])

    customer = {
        "Gender": gender, "Senior_Citizen": senior, "Partner": partner, "Dependents": dependents,
        "Tenure_Months": tenure, "Phone_Service": phone, "Multiple_Lines": multi,
        "Internet_Service": internet, **addons,
        "Contract": contract, "Paperless_Billing": paperless, "Payment_Method": payment,
        "Monthly_Charges": monthly, "Total_Charges": total,
    }

    with right:
        cut = st.session_state.cut
        p = float(predict(pd.DataFrame([customer]))[0])
        hot = p >= cut
        whole, dec = f"{p * 100:.1f}".split(".")
        verdict = "Contact this customer" if hot else ("Watch, no call yet" if p >= 0.30 else "Low risk, no action")
        st.markdown(f"""
<div class="lbl">Churn probability</div>
<div class="big {'hot' if hot else ''}">{whole}<small>.{dec}%</small></div>
<div class="line {'hot' if hot else ''}"><i style="width:{p * 100:.1f}%"></i><b style="left:{cut * 100:.1f}%"></b></div>
<div class="scale"><span>0</span><span>cut-off {cut:.2f}</span><span>1</span></div>
<div class="verdict {'hot' if hot else ''}">{verdict}</div>
<p class="note">{p / TRAIN_RATE:.1f}× the average churn rate of {pct(TRAIN_RATE)}.</p>
""", unsafe_allow_html=True)

        levers = []
        if contract == "Month-to-month":           # only offer longer contracts, never shorter ones
            levers.append(("One-year contract", {"Contract": "One year"}))
        if contract != "Two year":
            levers.append(("Two-year contract", {"Contract": "Two year"}))
        if internet != "No" and not (addons["Online_Security"] == "Yes" and addons["Tech_Support"] == "Yes"):
            levers.append(("Security + tech support", {"Online_Security": "Yes", "Tech_Support": "Yes"}))
        if "automatic" not in payment:
            levers.append(("Automatic card payment", {"Payment_Method": "Credit card (automatic)"}))
        if len(levers) > 1:
            combo = {}
            for _, ch in levers:
                combo.update(ch)
            combo["Contract"] = "Two year"
            levers.append(("All of these", combo))

        st.markdown('<div class="lbl">If we offered</div>', unsafe_allow_html=True)
        if levers:
            pv = predict(pd.DataFrame([{**customer, **ch} for _, ch in levers]))
            rows = "".join(
                f'<tr><td>{html.escape(name)}</td><td class="n {"hot" if q >= cut else ""}">{pct(q)}</td>'
                f'<td class="n g">{(q - p) * 100:+.1f}</td></tr>' for (name, _), q in zip(levers, pv))
            st.markdown(f'<table class="min">{rows}</table>', unsafe_allow_html=True)
            st.markdown('<p class="note">Model estimate for the same customer with that change. '
                        'A pattern from past customers, not proof of cause.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="note">Already on a two-year contract with protection and automatic payment.</p>',
                        unsafe_allow_html=True)

        st.slider("Contact when probability ≥", 0.05, 0.95, key="cut", step=0.01,
                  help="0.56 = best F1 on out-of-fold training predictions. Lower catches more churners and contacts more customers.")
        st.caption(f"0.56 best F1 · {bundle['threshold_recall_target']:.2f} for {bundle['recall_target']:.0%} recall · 0.50 default")

# ============================================================== TAB 2: batch
with tab_batch:
    cut = st.session_state.cut
    st.markdown('<p class="note" style="max-width:68ch">Score many customers at once. Upload a CSV or Excel file with the columns of '
                'the original dataset; spaces or underscores in names both work, and extra columns such as Churn Score are ignored. '
                'If the file has a Churn Value column, the page also reports how many real churners were caught.</p>',
                unsafe_allow_html=True)
    up = st.file_uploader("Customer file", type=["csv", "xlsx"], label_visibility="collapsed")
    use_sample = st.toggle("Use 30 demo customers from the test set", value=up is None)

    df_in = None
    if up is not None and not use_sample:
        df_in = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    elif use_sample and os.path.exists(SAMPLE_PATH):
        df_in = pd.read_csv(SAMPLE_PATH)

    if df_in is not None:
        std = standardise_columns(df_in)
        missing = [c for c in RAW_INPUT_COLUMNS if c not in std.columns]
        if missing:
            st.error(f"The file is missing required column(s): {', '.join(missing)}")
        else:
            gaps = std[RAW_INPUT_COLUMNS].replace(r"^\s*$", np.nan, regex=True).isna().sum()
            gaps = gaps[gaps > 0]
            if len(gaps):
                st.caption("Blank values filled by the model's imputer: " + ", ".join(f"{k} ({v})" for k, v in gaps.items()))
            probs = predict(df_in)
            out = pd.DataFrame(index=df_in.index)
            out["CustomerID"] = std["CustomerID"] if "CustomerID" in std.columns else [f"row {i + 1}" for i in range(len(std))]
            out["Churn probability"] = probs
            out["Decision"] = np.where(probs >= cut, "Contact", "No action")
            if "Churn_Value" in std.columns:
                out["Actual churn"] = std["Churn_Value"].astype(int)
            out = out.sort_values("Churn probability", ascending=False).reset_index(drop=True)
            out.insert(0, "Rank", range(1, len(out) + 1))

            n_flag = int((probs >= cut).sum())
            st.markdown(f'<div class="figs"><div><b>{len(out)}</b><span>customers scored</span></div>'
                        f'<div><b>{n_flag}</b><span>to contact</span></div>'
                        f'<div><b>{n_flag / len(out):.0%}</b><span>of the file</span></div></div>', unsafe_allow_html=True)
            if "Actual churn" in out:
                y = out["Actual churn"].to_numpy()
                yhat = (out["Churn probability"].to_numpy() >= cut).astype(int)
                tp, fp, fn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
                st.markdown(f'<p class="note">Real outcomes in the file: {tp} of {tp + fn} churners caught, {fp} loyal customers '
                            f'contacted unnecessarily, at cut-off {cut:.2f}. A sample this small is noisy; the 1,409-customer '
                            'test set is the reliable measure.</p>', unsafe_allow_html=True)
            rows = "".join(
                f'<tr><td class="g">{r.Rank}</td><td style="font-family:Geist Mono,monospace;font-size:.82rem">{html.escape(str(r.CustomerID))}</td>'
                f'<td class="n {"hot" if r._3 >= cut else ""}">{pct(r._3)}</td><td>{"Contact" if r._3 >= cut else "–"}</td>'
                + (f'<td class="g">{"yes" if r._5 else "no"}</td>' if "Actual churn" in out else "") + "</tr>"
                for r in out.itertuples())
            head = ('<tr><th>#</th><th>Customer</th><th class="n">Probability</th><th>Action</th>'
                    + ("<th>Churned</th>" if "Actual churn" in out else "") + "</tr>")
            st.markdown(f'<table class="min">{head}{rows}</table>', unsafe_allow_html=True)
            st.write("")
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")

# ============================================================== TAB 3: model card
with tab_card:
    a, _, b = st.columns([1, 0.1, 1])
    bp = bundle["best_params"]
    with a:
        st.markdown("### Model")
        facts = [
            ("Algorithm", "XGBoost, gradient-boosted trees"),
            ("Pipeline", "Clean, impute, scale, one-hot, then the trees"),
            ("Trained on", f"{bundle['train_rows']:,} customers (80%, stratified)"),
            ("Selected by", f"Tuned 5-fold CV ROC-AUC {bundle['cv_roc_auc']:.3f}, best of 7"),
            ("Cut-off", f"{bundle['threshold']:.2f}, best F1 on out-of-fold predictions"),
            ("Settings", f"{bp['n_estimators']} trees, depth {bp['max_depth']}, learning rate {bp['learning_rate']:.3f}, "
                         f"subsample {bp['subsample']:.2f}, colsample {bp['colsample_bytree']:.2f}, L2 {bp['reg_lambda']:.2f}"),
            ("Libraries", f"scikit-learn {saved_v['scikit-learn']}, xgboost {saved_v['xgboost']}"),
        ]
        st.markdown('<table class="min">' + "".join(f'<tr><td class="g" style="width:8rem">{k}</td><td>{v}</td></tr>' for k, v in facts)
                    + "</table>", unsafe_allow_html=True)
        st.markdown("### What the model relies on (drop in test ROC-AUC when shuffled)")
        drivers = sorted(bundle["top_drivers"].items(), key=lambda kv: -kv[1])
        mx = drivers[0][1]
        st.markdown("".join(f'<div class="bar"><span>{k.replace("_", " ")}</span><span><i style="width:{v / mx * 100:.1f}%"></i></span>'
                            f"<span>{v:.3f}</span></div>" for k, v in drivers), unsafe_allow_html=True)
    with b:
        st.markdown(f"### Test set, {bundle['test_rows']:,} unseen customers")
        m56, m50 = bundle["test_metrics"], bundle["test_metrics_default_0_5"]
        st.markdown('<table class="min"><tr><th></th><th class="n">Cut-off 0.56</th><th class="n">0.50</th></tr>'
                    + "".join(f'<tr><td>{k}</td><td class="n">{m56[k]:.3f}</td><td class="n g">{m50[k]:.3f}</td></tr>' for k in m56)
                    + "</table>", unsafe_allow_html=True)
        st.markdown("### Limits")
        limits = [
            "Precision is about 0.54 at 0.56: roughly half of contacted customers would have stayed, so offers should be cheap.",
            "Trained on IBM's fictional Californian telecom sample; new plans or prices need retraining.",
            "XGBoost, Random Forest and Logistic Regression were statistically tied on test ROC-AUC.",
            "What-if results are associations, not causal effects.",
        ]
        st.markdown('<table class="min">' + "".join(f"<tr><td>{x}</td></tr>" for x in limits) + "</table>", unsafe_allow_html=True)

st.markdown('<div class="foot">Pitti Prince Charles · 251090700046 · MCA 5141 ML Lab, MIT Manipal · '
            'IBM Telco Customer Churn, 7,043 customers · XGBoost</div>', unsafe_allow_html=True)
