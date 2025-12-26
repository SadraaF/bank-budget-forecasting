import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# --- SETUP ---
df = pd.read_csv('neobank_budget_data.csv')

# Feature Engineering
df['Prev_Month_Budget'] = df['Operational_Budget_B_Rials'].shift(1)
df['Prev_Month_USD'] = df['USD_Rate_Tomans'].shift(1)
df['User_Trend'] = df['Active_Users'].diff()
df = df.dropna().reset_index(drop=True)

features = ['Active_Users', 'USD_Rate_Tomans', 'Prev_Month_Budget']
target = 'Operational_Budget_B_Rials'
X = df[features]
y = df[target]

# ==========================================
# GRAPH 1: BACKTESTING (2025 Validation)
# ==========================================

# Train only on 2021-2024
split_idx = 48 
X_train_backtest = X.iloc[:split_idx].copy()
y_train_backtest = y.iloc[:split_idx].copy()

# Hybrid Model (Train on Past)
trend_model_bt = LinearRegression()
trend_model_bt.fit(X_train_backtest, y_train_backtest)
residuals_bt = y_train_backtest - trend_model_bt.predict(X_train_backtest)

xgb_model_bt = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model_bt.fit(X_train_backtest, residuals_bt)

def predict_bt(row):
    return trend_model_bt.predict(row) + xgb_model_bt.predict(row)

# Simulate 2025
start_row = df.iloc[split_idx-1]
dates_2025 = pd.to_datetime(df.iloc[split_idx:]['Date'])
dates_history = pd.to_datetime(df.iloc[:split_idx]['Date'])

scenarios_2025 = {
    'Optimistic': 0.01,
    'Moderate': 0.04,     
    'Pessimistic': 0.10      
}

results_2025 = {}

for name, infl_rate in scenarios_2025.items():
    curr_budget = start_row['Operational_Budget_B_Rials']
    curr_usd = start_row['USD_Rate_Tomans']
    curr_users = start_row['Active_Users']
    preds = []
    
    for i in range(len(dates_2025)):
        next_usd = curr_usd * (1 + infl_rate)
        next_users = curr_users + 150_000 
        
        row = pd.DataFrame([[next_users, next_usd, curr_budget]], columns=features)
        next_budget = predict_bt(row)[0]
        if next_budget < curr_budget: next_budget = curr_budget
        preds.append(next_budget)
        
        curr_budget = next_budget
        curr_usd = next_usd
        curr_users = next_users
    results_2025[name] = preds

# Plot Graph 1
plt.figure(figsize=(14, 7))
plt.plot(dates_history, y_train_backtest, label='History (2021-2024)', color='gray', linewidth=2)

colors = {'Optimistic': 'green', 'Moderate': 'black', 'Pessimistic': 'red'}
for name, preds in results_2025.items():
    plt.plot(dates_2025, preds, label=f'Predicted {name}', color=colors[name], linestyle='--')

plt.plot(dates_2025, df.iloc[split_idx:]['Operational_Budget_B_Rials'], label='Reality (2025)', color='blue', linewidth=4)

plt.title('Backtesting 2025', fontsize=16)
plt.ylabel('Budget (Billion Rials)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 2: FORECASTING 2026 (The Future)
# ==========================================

# RETRAIN on FULL DATA (2021-2025)
trend_model_full = LinearRegression()
trend_model_full.fit(X, y)
residuals_full = y - trend_model_full.predict(X)

xgb_model_full = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model_full.fit(X, residuals_full)

def predict_full(row):
    return trend_model_full.predict(row) + xgb_model_full.predict(row)

# Simulate 2026
last_row = df.iloc[-1]
dates_2026 = [pd.to_datetime(last_row['Date']) + pd.DateOffset(months=i+1) for i in range(12)]

# Scenarios for 2026
scenarios_2026 = {
    'Optimistic (Recovery)': 0.01,   
    'Realistic (High Inflation)': 0.04, 
    'Pessimistic (Hyper-Inflation)': 0.10
}

results_2026 = {}

for name, infl_rate in scenarios_2026.items():
    curr_budget = last_row['Operational_Budget_B_Rials']
    curr_usd = last_row['USD_Rate_Tomans']
    curr_users = last_row['Active_Users']
    preds = []
    
    for i in range(12):
        next_usd = curr_usd * (1 + infl_rate)
        next_users = curr_users + 150_000 
        
        row = pd.DataFrame([[next_users, next_usd, curr_budget]], columns=features)
        next_budget = predict_full(row)[0]
        if next_budget < curr_budget: next_budget = curr_budget
        preds.append(next_budget)
        
        curr_budget = next_budget
        curr_usd = next_usd
        curr_users = next_users
    results_2026[name] = preds

# Plot Graph 2
plt.figure(figsize=(14, 7))

# Plot Full History (2021-2025)
plt.plot(pd.to_datetime(df['Date']), y, label='History (2021-2025)', color='black', linewidth=2)

# Plot 2026 Scenarios
colors_26 = {'Optimistic (Recovery)': 'green', 'Realistic (High Inflation)': 'blue', 'Pessimistic (Hyper-Inflation)': 'red'}
for name, preds in results_2026.items():
    plt.plot(dates_2026, preds, label=f'2026 Forecast: {name}', color=colors_26[name], linestyle='--', marker='o', markersize=4)

plt.title('Budget Forecast For 2026', fontsize=16)
plt.ylabel('Budget (Billion Rials)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

joblib.dump(trend_model_full, 'model_trend.pkl')
joblib.dump(xgb_model_full, 'model_residual.pkl')