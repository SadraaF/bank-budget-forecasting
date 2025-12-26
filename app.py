import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import datetime

# ==========================================
# 0. HELPER: PERSIAN NUMERALS
# ==========================================
def to_persian_num(text):
    """Converts English digits to Persian digits."""
    text = str(text)
    mapping = {
        '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
        '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
        '.': '/', ',': '،'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

# ==========================================
# 1. PAGE CONFIG & LAYOUT FIXES
# ==========================================
st.set_page_config(
    page_title="سامانه پیش‌بینی بودجه",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@100..900&display=swap');

    .stApp {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    
    *:not(.material-icons):not([data-testid="stSidebarCollapseButton"] *) {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-right: 2.5rem !important; 
        padding-left: 1rem !important;
    }
    
    section[data-testid="stSidebar"] {
        text-align: right !important;
        direction: rtl; 
    }
    
    .stSlider {
        direction: ltr; 
        padding-top: 20px; 
    }
    
    .stSlider label {
        width: 100%;
        text-align: right !important;
        direction: rtl;
        font-size: 1rem !important;
        font-weight: bold;
    }

    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        pointer-events: none !important;
    }
    .anchor-link, a.anchor-link {
        display: none !important;
        visibility: hidden !important;
    }

    .stMarkdown, .stText, p, div {
        text-align: right;
    }
    
    .modebar { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOGIC
# ==========================================
@st.cache_resource
def load_and_train_model():
    np.random.seed(777)
    MONTHS = 60
    dates = [datetime.date(2021, 1, 1) + datetime.timedelta(days=30*i) for i in range(MONTHS)]
    df = pd.DataFrame({'Date': dates, 'Month_Index': range(MONTHS)})
    
    xp = [0, 15, 22, 26, 30, 38, 40, 44, 48, 54, 59]
    fp = [23000, 27000, 36000, 59000, 48000, 53000, 65000, 58000, 72000, 105000, 134975]
    base_usd = np.interp(df['Month_Index'], xp, fp)
    df['USD_Rate_Tomans'] = (base_usd + np.random.normal(0, 1000, MONTHS)).astype(int)
    
    df['Active_Users'] = (9_000_000 / (1 + np.exp(-0.13 * (df['Month_Index'] - 38)))).astype(int)
    
    hedged_usd = df['USD_Rate_Tomans'].rolling(6, min_periods=1).max()
    cost_infra = df['Active_Users'] * 0.35 * (hedged_usd * 10)
    
    ops_vals = []
    curr_ops = 90_000_000_000 
    for i in range(MONTHS):
        target = 90e9 * (hedged_usd[i]/23000) * (1 + df.loc[i, 'Active_Users']/10e6)
        if target > curr_ops: curr_ops = target
        ops_vals.append(curr_ops)
        
    cost_mkt = df['Active_Users'].diff().fillna(1000).clip(lower=0) * 4.0 * (df['USD_Rate_Tomans'] * 10)
    raw_total = (cost_infra + cost_mkt + np.array(ops_vals)) / 1_000_000_000
    df['Operational_Budget_B_Rials'] = pd.Series(raw_total).cummax().round(2)
    
    df['Prev_Month_Budget'] = df['Operational_Budget_B_Rials'].shift(1)
    df['Prev_Month_USD'] = df['USD_Rate_Tomans'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    features = ['Active_Users', 'USD_Rate_Tomans', 'Prev_Month_Budget']
    target = 'Operational_Budget_B_Rials'
    X = df[features]
    y = df[target]
    
    trend_model = LinearRegression()
    trend_model.fit(X, y)
    residuals = y - trend_model.predict(X)
    xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, residuals)
    
    return df, trend_model, xgb_model, features

df_hist, trend_model, xgb_model, features = load_and_train_model()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: right; direction: rtl; color: #FFFFFF; font-size: 1.2rem; font-weight: bold; margin-bottom: 10px;">
    ⚙️ تنظیمات شبیه‌سازی
    """, unsafe_allow_html=True)

    st.markdown("### متغیرهای اقتصادی")
    inflation_input = st.slider(
        "نرخ تورم ماهانه (درصد)", 
        0.0, 15.0, 3.5, 0.5
    )

    st.markdown("### متغیرهای کسب‌وکار")
    growth_input = st.slider(
        "رشد ماهانه کاربران", 
        10_000, 500_000, 150_000, 10_000
    )

    st.markdown("---")
    st.markdown("👨‍💻 **پروژه بانکداری توسط:** تینا شاکریان، فاطمه‌سادات قائمی، صدرا فدائی، صبا داوودآبادی، سانیا عزتی")

# ==========================================
# 4. SIMULATION
# ==========================================
def run_simulation(inflation_pct, user_growth):
    last_row = df_hist.iloc[-1]
    future_dates = [pd.to_datetime(last_row['Date']) + pd.DateOffset(months=i+1) for i in range(12)]
    
    curr_budget = last_row['Operational_Budget_B_Rials']
    curr_usd = last_row['USD_Rate_Tomans']
    curr_users = last_row['Active_Users']
    
    preds = []
    usds = []
    
    for i in range(12):
        next_usd = curr_usd * (1 + inflation_pct/100)
        next_users = curr_users + user_growth
        
        row = pd.DataFrame([[next_users, next_usd, curr_budget]], columns=features)
        trend = trend_model.predict(row)[0]
        resid = xgb_model.predict(row)[0]
        next_budget = max(trend + resid, curr_budget)
        
        preds.append(next_budget)
        usds.append(next_usd)
        
        curr_budget = next_budget
        curr_usd = next_usd
        curr_users = next_users
        
    return future_dates, preds, usds

future_dates, predicted_budget, predicted_usd = run_simulation(inflation_input, growth_input)

# ==========================================
# 5. UI COMPONENTS
# ==========================================
st.markdown("""
<h1 style='text-align: right; color: white; font-family: Vazirmatn;'>سامانه مدیریت ریسک نقدینگی نئوبانک 🏦</h1>
<h4 style='text-align: right; color: #cccccc; font-family: Vazirmatn;'>پیش‌بینی بودجه عملیاتی سال ۲۰۲۶ مبتنی بر هوش مصنوعی</h4>
<hr>
""", unsafe_allow_html=True)

val_budget_curr = int(df_hist.iloc[-1]['Operational_Budget_B_Rials'])
val_budget_pred = int(predicted_budget[-1])
val_increase = int(val_budget_pred - val_budget_curr)
val_usd = int(predicted_usd[-1])

p_budget_curr = to_persian_num(f"{val_budget_curr:,}")
p_budget_pred = to_persian_num(f"{val_budget_pred:,}")
p_increase = to_persian_num(f"{val_increase:,}")
p_usd = to_persian_num(f"{val_usd:,}")

card_style = """
    background-color: #1E2129;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #444;
    text-align: right;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    color: white;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    font-family: 'Vazirmatn', sans-serif;
"""

c1, c2, c3, c4 = st.columns(4)

with c4:
    st.markdown(f"""
    <div style="{card_style} border-right: 4px solid #4B90FF;">
        <span style="color: #aaa; font-size: 0.9em;">بودجه فعلی ماهانه</span>
        <span style="font-size: 1.8em; font-weight: bold; margin: 5px 0;">{p_budget_curr}</span>
        <span style="color: #aaa; font-size: 0.8em;">میلیارد ریال</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="{card_style} border-right: 4px solid #4B90FF;">
        <span style="color: #aaa; font-size: 0.9em;">پیش‌بینی پایان ۲۰۲۶</span>
        <span style="font-size: 1.8em; font-weight: bold; margin: 5px 0;">{p_budget_pred}</span>
        <span style="color: #FF4B4B; background: rgba(255,75,75,0.2); padding: 2px 6px; border-radius: 4px; font-size: 0.8em; width: fit-content;">+{p_increase} افزایش</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div style="{card_style} border-right: 4px solid #00D26A;">
        <span style="color: #aaa; font-size: 0.9em;">بافر نقدینگی مورد نیاز</span>
        <span style="font-size: 1.8em; font-weight: bold; margin: 5px 0;">{p_increase}</span>
        <span style="color: #aaa; font-size: 0.8em;">میلیارد ریال</span>
    </div>
    """, unsafe_allow_html=True)

with c1:
    st.markdown(f"""
    <div style="{card_style} border-right: 4px solid #FFC107;">
        <span style="color: #aaa; font-size: 0.9em;">نرخ دلار پیش‌بینی شده</span>
        <span style="font-size: 1.8em; font-weight: bold; margin: 5px 0;">{p_usd}</span>
        <span style="color: #aaa; font-size: 0.8em;">تومان</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""<br>
<h3>نمودار تحلیل 📉</h3>""", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.to_datetime(df_hist['Date']), 
    y=df_hist['Operational_Budget_B_Rials'],
    mode='lines',
    name='گذشته',
    line=dict(color='#A3A8B8', width=2)
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=predicted_budget,
    mode='lines+markers',
    name='پیش‌بینی',
    line=dict(color='#FF4B4B', width=3),
    marker=dict(size=8, color='#FF4B4B')
))

tick_dates = pd.to_datetime(df_hist['Date'].tolist() + future_dates)
tick_vals = tick_dates[::6] 
tick_text = [to_persian_num(d.strftime('%Y-%m')) for d in tick_vals]

max_val = max(max(df_hist['Operational_Budget_B_Rials']), max(predicted_budget))
y_tick_vals = list(range(0, int(max_val)+2000, 2000))
y_tick_text = [to_persian_num(str(val)) for val in y_tick_vals]

fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        title='تاریخ',
        showgrid=True, 
        gridcolor='#262730'
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=y_tick_vals,
        ticktext=y_tick_text,
        title='بودجه (میلیارد ریال)',
        title_standoff=60, # Keep the standoff
        showgrid=True, 
        gridcolor='#262730'
    ),
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(family="Vazirmatn", color="white"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    margin=dict(l=80, r=20, t=50, b=50) 
)

st.plotly_chart(fig, use_container_width=True)