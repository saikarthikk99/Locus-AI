# Crime Hotspot Detection & Crime Prediction
**Amrita Vishwa Vidyapeetham**

**Team:** Rahul Anurag Sai, T. Sathvika, T. Sri Nikesh, V V S Karthik  


---

## Quick Start

```bash
# 1. Install dependencies
pip install flask pandas scikit-learn folium werkzeug

# 2. Run the app
python app.py

# 3. Open browser
http://127.0.0.1:5000
```

---

## Default Credentials

| Username | Password | Role  |
|----------|----------|-------|
| admin    | admin123 | Admin |
| rahul    | crime    | Team  |
| sats     | crime    | Team  |
| nikesh   | crime    | Team  |
| karthik  | crime    | Team  |

**Officer Registration Key:** `police@2025`

---

## Project Structure

```
MFC/
├── app.py                  ← Flask app + ML logic (DBSCAN + Random Forest)
├── db.py                   ← SQLite auth (register/login)
├── crime_data.db           ← Auto-created on first run
├── uploads/
│   └── crime_dataset.csv   ← Uploaded CSV stored here
├── static/
│   └── crime_map_embed.html ← Folium map (auto-generated)
└── templates/
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── upload.html
    ├── map.html
    └── predict.html
```

---

## CSV Format Required

Your CSV must have these columns (exact names selectable on upload):

| Column       | Options                          |
|--------------|----------------------------------|
| Latitude     | latitude, lat, Latitude, LAT     |
| Longitude    | longitude, lon, Longitude, LON   |
| City         | city, City, CITY, location       |
| Crime Type   | crime_type, CrimeType, type, category, crime |

Optional columns that improve predictions: `hour`, `day_of_week`, `month`

---

## How It Works

1. **Upload CSV** → System loads data, drops null rows
2. **DBSCAN** → Clusters crime coordinates using haversine distance  
   - `eps=0.01` (~1km radius), `min_samples=3`
   - Labels: `-1` = noise/isolated, `0+` = cluster ID
3. **Risk Classification** → `high` (≥10), `medium` (5–9), `low` (<5)
4. **Random Forest** → Trains 100-tree classifier with PCA reduction  
   - Features: lat, lon, cluster label, hour, day (if available)
5. **Folium Map** → CartoDB dark map with color-coded cluster markers
6. **Predict** → Input coordinates + time → predicted crime type + confidence

---

## Tech Stack

- **Backend:** Python 3, Flask, Pandas, Scikit-learn, SQLite
- **ML:** DBSCAN (sklearn), Random Forest (sklearn), PCA (sklearn)
- **Map:** Folium (CartoDB dark tiles)
- **Auth:** Werkzeug password hashing, Flask sessions
- **Frontend:** Pure HTML/CSS (no frameworks), IBM Plex Mono + Bebas Neue
