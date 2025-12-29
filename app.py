import os
import io
import joblib
import pandas as pd
import requests
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "3264b3b1d5ca2b7f24b7c1556728a85d")  # keep your key or set in .env
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")  # optional secondary provider

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DATA & MODEL ----------
DATA_FILE = "soil_recommendations_expanded.xlsx"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in project folder.")
df = pd.read_excel(DATA_FILE)

MODEL_PATH = "crop_recommender_rf.joblib"
model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)
except Exception as e:
    print("Model load failed — falling back to rule-based recommender.", e)
    model = None

# ---------- CROP / FERTILIZER INFO ----------
OPTIMAL_TEMP = {
    "Paddy": (20, 35), "Rice": (20, 35), "Maize": (18, 32),
    "Cotton": (20, 38), "Groundnut": (20, 35), "Sorghum": (25, 40),
    "Millets": (25, 42), "Chilli": (20, 32), "Sunflower": (20, 35),
    "Vegetables": (15, 30),
}
BASE_FERT = {
    "Paddy": {"N":100, "P2O5":50, "K2O":40},
    "Maize": {"N":120, "P2O5":60, "K2O":40},
    "Cotton": {"N":80, "P2O5":50, "K2O":50},
    "Groundnut": {"N":20, "P2O5":60, "K2O":40},
    "Sorghum": {"N":40, "P2O5":40, "K2O":20},
    "Millets": {"N":30, "P2O5":30, "K2O":20},
    "Chilli": {"N":80, "P2O5":50, "K2O":50},
    "Sunflower": {"N":60, "P2O5":50, "K2O":40},
    "Vegetables": {"N":100, "P2O5":60, "K2O":60},
    "Local_crop": {"N":50, "P2O5":30, "K2O":20}
}

# ---------- HELPERS ----------
def geocode_openweather(block, district):
    """Try OpenWeather geocoding."""
    if not OPENWEATHER_KEY:
        return None
    try:
        q = f"{block},{district},IN"
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={requests.utils.quote(q)}&limit=1&appid={OPENWEATHER_KEY}"
        r = requests.get(url, timeout=8).json()
        if r:
            return r[0]["lat"], r[0]["lon"]
        # fallback to district only
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={requests.utils.quote(district + ',IN')}&limit=1&appid={OPENWEATHER_KEY}"
        r = requests.get(url, timeout=8).json()
        if r:
            return r[0]["lat"], r[0]["lon"]
    except Exception as e:
        print("OpenWeather geocode error:", e)
    return None

def geocode(block, district):
    # primary: try OpenWeather geocode; if fails, try dataset lat/lon columns if present
    coords = geocode_openweather(block or "", district or "")
    if coords:
        return coords
    # try dataset lat/lon if available
    try:
        sub = df[(df['District'].astype(str) == str(district)) & (df['Block'].astype(str) == str(block))]
        if not sub.empty and 'Latitude' in sub.columns and 'Longitude' in sub.columns:
            row = sub.iloc[0]
            return float(row['Latitude']), float(row['Longitude'])
    except Exception:
        pass
    return None

def get_temperature_from_openweather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
        r = requests.get(url, timeout=8).json()
        return r.get("main", {}).get("temp")
    except Exception as e:
        print("OpenWeather temp error:", e)
        return None

def get_temperature_from_weatherapi(lat, lon):
    if not WEATHERAPI_KEY:
        return None
    try:
        q = f"{lat},{lon}"
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={requests.utils.quote(q)}"
        r = requests.get(url, timeout=8).json()
        return r.get("current", {}).get("temp_c")
    except Exception as e:
        print("WeatherAPI error:", e)
        return None

def get_temperature(lat, lon):
    """Try OpenWeather -> WeatherAPI -> None"""
    if lat is None or lon is None:
        return None
    t = get_temperature_from_openweather(lat, lon)
    if t is not None:
        return t
    t = get_temperature_from_weatherapi(lat, lon)
    return t

def rule_recommend_crop(soil_row, temp):
    n = soil_row.get('N') or soil_row.get('Nitrogen') or soil_row.get('n') or 0
    p = soil_row.get('P') or soil_row.get('Phosphorus') or soil_row.get('p') or 0
    k = soil_row.get('K') or soil_row.get('Potassium') or soil_row.get('k') or 0
    try:
        if temp is not None:
            for crop, (lo, hi) in OPTIMAL_TEMP.items():
                if lo <= temp <= hi:
                    return crop
        s = sum([float(x or 0) for x in [n, p, k]])
        if s >= 250:
            return "Maize"
        elif s >= 150:
            return "Paddy"
        else:
            return "Millets"
    except Exception:
        return "Local_crop"

def compute_fertilizer(crop, soil_row, area_acres):
    base = BASE_FERT.get(crop, BASE_FERT['Local_crop'])
    mult_map = {"Low":1.0, "Medium":0.7, "High":0.5, None:0.8}
    N = round(base['N'] * mult_map.get(soil_row.get('N_status'), 0.8) * area_acres, 1)
    P2O5 = round(base['P2O5'] * mult_map.get(soil_row.get('P_status'), 0.8) * area_acres, 1)
    K2O = round(base['K2O'] * mult_map.get(soil_row.get('K_status'), 0.8) * area_acres, 1)
    urea_kg = round((N / 0.46) if N>0 else 0, 1)
    ssp_kg = round((P2O5 / 0.16) if P2O5>0 else 0, 1)
    mop_kg = round((K2O / 0.60) if K2O>0 else 0, 1)
    return {"N_kg": N, "P2O5_kg": P2O5, "K2O_kg": K2O,
            "urea_kg": urea_kg, "ssp_kg": ssp_kg, "mop_kg": mop_kg}

def determine_condition(farmer_crop, recommended_crop, temperature):
    if temperature is None:
        return "Abnormal (no temperature)"
    if farmer_crop in OPTIMAL_TEMP:
        lo, hi = OPTIMAL_TEMP[farmer_crop]
        if lo <= temperature <= hi:
            return "Normal" if farmer_crop == recommended_crop else "Abnormal"
        else:
            return "Dangerous"
    else:
        if recommended_crop in OPTIMAL_TEMP:
            lo, hi = OPTIMAL_TEMP[recommended_crop]
            return "Abnormal" if lo <= temperature <= hi else "Dangerous"
        return "Abnormal"

def extract_nutrients(soil_row):
    out = {}
    if soil_row is None:
        return out
    for col, val in soil_row.items():
        if pd.notna(val):
            out[col] = val
    return out

# ---------- GRAPH ROUTE ----------
@app.route("/nutrients_graph")
def nutrients_graph():
    district = request.args.get("district")
    block = request.args.get("block")
    if not district or not block:
        return ("Missing district/block", 400)

    sub = df[(df['District'].astype(str) == str(district)) & (df['Block'].astype(str) == str(block))]
    if sub.empty:
        # try district-level
        sub = df[df['District'].astype(str) == str(district)]
        if sub.empty:
            return ("No data", 404)
    row = sub.iloc[0].to_dict()

    # pick numeric nutrient columns for plotting
    numeric_items = {}
    for k, v in row.items():
        # ignore District/Block and lat/lon
        if k.lower() in ('district','block','latitude','longitude'):
            continue
        try:
            fv = float(v)
            numeric_items[k] = fv
        except Exception:
            continue

    if not numeric_items:
        return ("No numeric nutrients to plot", 404)

    # Create bar chart
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8,4))
    items = list(numeric_items.items())
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title(f"Nutrients — {block}, {district}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    return send_file(buf, mimetype='image/png', download_name='nutrients.png')

# ---------- API ROUTES ----------
@app.route("/")
def index():
    df_local = df.copy()
    df_local['District'] = df_local['District'].astype(str)
    df_local['Block'] = df_local['Block'].astype(str)
    blocks_by_district = df_local.groupby('District')['Block'].unique().to_dict()
    blocks_by_district = {str(k): [str(x) for x in v] for k,v in blocks_by_district.items()}
    districts = sorted(list(blocks_by_district.keys()))
    crops = sorted(list(OPTIMAL_TEMP.keys()))
    return render_template("index.html", districts=districts, blocks_by_district=blocks_by_district, crops=crops)

@app.route("/get_nutrients")
def get_nutrients():
    district = request.args.get("district")
    block = request.args.get("block")
    if not district or not block:
        return jsonify({"nutrients": None})

    sub = df[(df['District'].astype(str) == str(district)) & (df['Block'].astype(str) == str(block))]
    if sub.empty:
        sub = df[df['District'].astype(str) == str(district)]
        if sub.empty:
            return jsonify({"nutrients": None})

    row = sub.iloc[0].to_dict()
    nutrients = extract_nutrients(row)
    return jsonify({"nutrients": nutrients})

@app.route("/get_recommendation", methods=["POST"])
def get_recommendation():
    data = request.form
    district = data.get("district")
    block = data.get("block")
    farmer_crop = data.get("crop")
    # user-provided temperature (optional)
    user_temp = data.get("temperature")
    # soil moisture from user (optional)
    user_moisture = data.get("moisture")
    try:
        area = float(data.get("area", "1"))
        if area <= 0:
            area = 1.0
    except:
        area = 1.0

    # Find soil row
    soil_row = None
    sub = df[(df['District'].astype(str) == str(district)) & (df['Block'].astype(str) == str(block))]
    if not sub.empty:
        soil_row = sub.iloc[0].to_dict()
    else:
        sub = df[df['District'].astype(str) == str(district)]
        if not sub.empty:
            soil_row = sub.iloc[0].to_dict()

    # Get coordinates (try geocode or dataset lat/lon)
    coords = geocode(block or "", district or "")
    temp = None
    if user_temp:
        try:
            temp = float(user_temp)
        except:
            temp = None
    else:
        if coords:
            temp = get_temperature(coords[0], coords[1])

    # If still None, temp remains None — frontend will allow manual input
    recommended = None
    if model is not None and soil_row is not None:
        feat = [soil_row.get(f, 0) for f in ['N','P','K','pH','OC','EC']]
        # ensure length is OK for model — append temp fallback
        feat.append(temp if temp is not None else 25)
        try:
            predicted = model.predict([feat])[0]
            # if model returns numeric label, keep fallback
            recommended = predicted if isinstance(predicted, str) else rule_recommend_crop(soil_row or {}, temp)
        except Exception as e:
            print("Model predict failed:", e)
            recommended = rule_recommend_crop(soil_row or {}, temp)
    else:
        recommended = rule_recommend_crop(soil_row or {}, temp)

    ferts = compute_fertilizer(recommended, soil_row or {}, area)
    condition = determine_condition(farmer_crop, recommended, temp)
    nutrients_out = extract_nutrients(soil_row)

    resp = {
        "district": district,
        "block": block,
        "temperature": temp,                # can be None
        "farmer_crop": farmer_crop,
        "recommended_crop": recommended,
        "condition": condition,
        "fertilizer_kg": ferts,
        "fertilizer_products_kg": {        # keep for backward compat if any front-end used it earlier
            "urea_kg": ferts["urea_kg"],
            "ssp_kg": ferts["ssp_kg"],
            "mop_kg": ferts["mop_kg"]
        },
        "soil_moisture": float(user_moisture) if user_moisture else None,
        "nutrients": nutrients_out
    }
    return jsonify(resp)

if __name__ == "__main__":
    # If running in dev, allow debug True. For production set debug=False.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
