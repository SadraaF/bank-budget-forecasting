import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

np.random.seed(777) 
MONTHS = 60 
START_DATE = datetime.date(2021, 1, 1)

def generate_neobank_data():
    dates = [START_DATE + datetime.timedelta(days=30*i) for i in range(MONTHS)]
    df = pd.DataFrame({'Date': dates, 'Month_Index': range(MONTHS)})
    
    # 1. USD MAPPING
    xp = [0, 15, 22, 26, 30, 38, 40, 44, 48, 54, 59]
    fp = [23000, 27000, 36000, 59000, 48000, 53000, 65000, 58000, 72000, 105000, 134975]
    base_usd = np.interp(df['Month_Index'], xp, fp)
    df['USD_Rate_Tomans'] = (base_usd + np.random.normal(0, 1000, MONTHS)).astype(int)

    # 2. BANK METRICS (Growth)
    # S-Curve for Users
    df['Active_Users'] = (9_000_000 / (1 + np.exp(-0.13 * (df['Month_Index'] - 38)))).astype(int)
    
    # 3. BUDGET COMPONENTS
    # Infrastructure: 0.35 USD per user (Hedged/Rolling Max)
    hedged_usd = df['USD_Rate_Tomans'].rolling(6, min_periods=1).max()
    cost_infra = df['Active_Users'] * 0.35 * (hedged_usd * 10)
    
    # Operations: Base salary mass that scales with users and inflation
    # But salaries ONLY go up.
    ops_vals = []
    curr_ops = 90_000_000_000 # 90B Rials base
    for i in range(MONTHS):
        # Pressure to increase: Inflation + Bank Size
        target = 90e9 * (hedged_usd[i]/23000) * (1 + df.loc[i, 'Active_Users']/10e6)
        if target > curr_ops: curr_ops = target
        ops_vals.append(curr_ops)
    
    new_users = df['Active_Users'].diff().fillna(1000).clip(lower=0)
    cost_mkt = new_users * 4.0 * (df['USD_Rate_Tomans'] * 10)

    # 4. TOTAL BUDGET WITH HARD RATCHET
    raw_total = (cost_infra + cost_mkt + np.array(ops_vals)) / 1_000_000_000
    df['Operational_Budget_B_Rials'] = pd.Series(raw_total).cummax().round(2)
    
    # Metadata
    df['Inflation_Rate'] = df['USD_Rate_Tomans'].pct_change(12).fillna(0) * 100
    df['Transaction_Volume'] = (df['Active_Users'] * 25).astype(int)

    return df

data = generate_neobank_data()
data.to_csv('neobank_budget_data.csv', index=False)