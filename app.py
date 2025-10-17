import json, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Load artifacts
# -----------------------------
ART_DIR = Path("./artifacts")
lin = joblib.load(ART_DIR / "linear_loglog.pkl")
cfg = json.load(open(ART_DIR / "linear_config.json"))

CAT_COL   = cfg["cat_cols"][0] if cfg.get("cat_cols") else "product_category_name"
NUM_LOGS  = cfg.get("num_logs", ["log_price"])
DEFAULTS  = cfg.get("feature_defaults", {})
CATS      = cfg.get("categories", ["bed_bath_table"])
P_BOUNDS  = cfg.get("price_bounds", {"p5": 20.0, "p95": 350.0})

def _src_from_log_name(log_name: str) -> str:
    return "price_cap" if log_name == "log_price" else log_name.replace("log_", "")

st.set_page_config(page_title="Retail Price Optimizer (Linear)", layout="wide")
st.title("Retail Price Optimization — Linear (log–log)")
st.caption("Minimal demo using the saved classical model. Provide a few inputs → see demand & profit curves.")

# -----------------------------
# Sidebar inputs (minimal)
# -----------------------------
st.sidebar.header("Inputs")
cat_val = st.sidebar.selectbox("Category", CATS, index=0)

p_min = float(P_BOUNDS.get("p5", 20.0))
p_max = float(P_BOUNDS.get("p95", 350.0))
price = st.sidebar.slider("Your price", min_value=p_min, max_value=p_max, value=min(120.0, p_max), step=0.5)
lag_price = st.sidebar.number_input("Lag price (prev)", value=float(DEFAULTS.get("lag_price", price)), step=0.5)

with st.sidebar.expander("Advanced (competitors & promos)"):
    comp_1 = st.number_input("comp_1", value=float(DEFAULTS.get("comp_1", lag_price)), step=0.5)
    comp_2 = st.number_input("comp_2", value=float(DEFAULTS.get("comp_2", lag_price)), step=0.5)
    comp_3 = st.number_input("comp_3", value=float(DEFAULTS.get("comp_3", lag_price)), step=0.5)
    ps1 = st.number_input("ps1", value=float(DEFAULTS.get("ps1", 4.0)), step=0.1)
    ps2 = st.number_input("ps2", value=float(DEFAULTS.get("ps2", 4.0)), step=0.1)
    ps3 = st.number_input("ps3", value=float(DEFAULTS.get("ps3", 4.0)), step=0.1)
    fp1 = st.number_input("fp1", value=float(DEFAULTS.get("fp1", 15.0)), step=0.1)
    fp2 = st.number_input("fp2", value=float(DEFAULTS.get("fp2", 20.0)), step=0.1)
    fp3 = st.number_input("fp3", value=float(DEFAULTS.get("fp3", 15.0)), step=0.1)
    year_ex  = st.number_input("year_ex",  value=int(DEFAULTS.get("year_ex", 2018)), step=1)
    month_ex = st.number_input("month_ex", value=int(DEFAULTS.get("month_ex", 1)), min_value=1, max_value=12, step=1)
    week_ex  = st.number_input("week_ex",  value=int(DEFAULTS.get("week_ex", 1)),  min_value=1, max_value=53, step=1)

# -----------------------------
# Build single-scenario row
# -----------------------------
raw = {
    "price_cap": price, "lag_price": lag_price,
    "comp_1": comp_1, "comp_2": comp_2, "comp_3": comp_3,
    "ps1": ps1, "ps2": ps2, "ps3": ps3,
    "fp1": fp1, "fp2": fp2, "fp3": fp3,
    "year_ex": year_ex, "month_ex": month_ex, "week_ex": week_ex,
    CAT_COL: cat_val,
}

X_one = pd.DataFrame([{CAT_COL: raw[CAT_COL]}])
for lname in NUM_LOGS:
    src = _src_from_log_name(lname)
    val = float(raw.get(src, DEFAULTS.get(src, 0.0)))
    X_one[lname] = np.log(max(val, 1e-9))

pred_qty = float(np.expm1(lin.predict(X_one))[0])

st.subheader("Predicted Demand")
st.markdown(
    f"<h2 style='text-align:center; color:#2C7BE5;'>Predicted Quantity: {pred_qty:,.2f} units</h2>",
    unsafe_allow_html=True
)

# -----------------------------
# Price Sensitivity (Demand & Profit)
# -----------------------------
st.subheader("Price Sensitivity Analysis")

grid = np.linspace(p_min, p_max, 60)
def _row_with_price(p: float):
    r = X_one.copy()
    r["log_price"] = np.log(max(float(p), 1e-9))
    return r

X_grid = pd.concat([_row_with_price(p) for p in grid], ignore_index=True)
q_pred = np.expm1(lin.predict(X_grid))

unit_cost = 0.6 * np.percentile(grid, 25)
profit = (grid - unit_cost) * q_pred
p_star = float(grid[np.argmax(profit)])

# -----------------------------
# Graphs (reduced size)
# -----------------------------
# --- Graphs (same heights) ---
c1, c2 = st.columns(2)

with c1:
    fig1, ax1 = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    ax1.plot(grid, q_pred, color="#007ACC")
    ax1.set_title("Demand Curve", fontsize=11)
    ax1.set_xlabel("Price"); ax1.set_ylabel("Predicted Qty")
    ax1.grid(True, alpha=0.25)
    st.pyplot(fig1, use_container_width=True)

with c2:
    fig2, ax2 = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    ax2.plot(grid, profit, color="#2E8B57")
    ax2.axvline(p_star, linestyle="--", color="red", label=f"Optimal ≈ {p_star:.2f}")
    ax2.set_title("Profit Curve", fontsize=11)
    ax2.set_xlabel("Price"); ax2.set_ylabel("Profit (units)")
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2, use_container_width=True)



st.caption(
    "Model: log–log linear (we model log(quantity+1) vs log(price) & other logs). "
    "Profit uses the  refrerenced unit cost — Business will replace with actual cost for real decisions."
)
