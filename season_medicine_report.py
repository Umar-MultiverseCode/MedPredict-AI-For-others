import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from xgboost import XGBRegressor
import joblib

app = Flask(__name__, static_folder='static')

# -------------------------------
# 1. Load Model & Encoders
# -------------------------------
model = joblib.load("medicine_demand_model.pkl")
le_name = joblib.load("le_name.pkl")
le_season = joblib.load("le_season.pkl")

# -------------------------------
# 2. Load and Preprocess Data
# -------------------------------
data = pd.read_csv("rural_clinic_medicines_dataset_updated.csv")
data.columns = data.columns.str.strip()

data['manufacture_date'] = pd.to_datetime(data['manufacture_date'], errors='coerce', dayfirst=True)
data['expiry_date'] = pd.to_datetime(data['expiry_date'], errors='coerce', dayfirst=True)
data.fillna({'stock_remaining':0, 'quantity':0}, inplace=True)

for col in ['name','manufacturer_name','type','pack_size_label','location_id','category','season']:
    data[col] = data[col].astype(str).str.strip()

def get_season(date):
    month = date.month
    if month in [3,4,5]:
        return 'Summer'
    elif month in [6,7,8,9]:
        return 'Monsoon'
    elif month in [10,11]:
        return 'Autumn'
    else:
        return 'Winter'

data['calculated_season'] = data['manufacture_date'].apply(get_season)

data['name_enc'] = le_name.transform(data['name'])
data['season_enc'] = le_season.transform(data['calculated_season'])

@app.route('/report')
def generate_report():
    user_season = request.args.get('season')
    if user_season not in ['Summer','Monsoon','Autumn','Winter']:
        return jsonify({'error': 'Invalid Season! Please enter Summer, Monsoon, Autumn, or Winter.'}), 400

    season_enc = le_season.transform([user_season])[0]

    # Filter medicines relevant for this season
    season_meds = data[data['calculated_season'] == user_season].copy()
    report = []

    for idx, row in season_meds.iterrows():
        name = row['name']
        name_enc = row['name_enc']
        
        predicted_demand = model.predict([[name_enc, season_enc]])[0]
        
        # Only include medicines where stock < predicted demand
        stock_remaining = row['stock_remaining']
        if stock_remaining >= predicted_demand:
            continue
        
        reorder_qty = max(0, predicted_demand - stock_remaining)
        
        report.append({
            'Medicine': name,
            'Predicted_Demand': round(predicted_demand),
            'Stock_Remaining': stock_remaining,
            'Reorder_Quantity': round(reorder_qty)
        })

    # Sort by urgency (highest reorder first)
    report_df = pd.DataFrame(report).sort_values(by='Reorder_Quantity', ascending=False)

    return jsonify(report_df.to_dict('records'))

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
