# app.py
# Crime Spot Detection — ALL PHASES COMBINED
# Team 05 | Amrita Vishwa Vidyapeetham | 22MAT122
#
# Phase 1: Auth, Upload, DBSCAN, Map, Random Forest, Dashboard
# Phase 2: City analysis, crime filtering, top risky cities, trend analysis
# Phase 3: Crime Probability Prediction + Crime Time Forecast
# Phase 4: Model Feedback Learning + Context-Aware Prediction + Explanation Engine
# Social Sieve: Mock social media crime pulse + Red Alert cluster detection

from flask import Flask, render_template, request, redirect, url_for, session, flash
from db import (create_tables, register_user, login_user,
                setup_default_admin, setup_default_users,
                create_feedback_table, save_feedback, get_feedback_count)
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "authentication_demo_secret"
app.config['UPLOAD_FOLDER'] = 'uploads'

OFFICER_SECRET_KEY = "police@2025"

# -----------------------------------------------
# Global state — lives in memory while server runs
# -----------------------------------------------
crime_df       = None   # loaded DataFrame
col_config     = {}     # column mapping: lat, lon, city, crime
cluster_stats  = {}     # DBSCAN cluster info per cluster ID
trained_model  = None   # Random Forest classifier
label_encoder  = None   # encodes/decodes crime type labels
pca_model      = None   # PCA fitted on training data (no leakage)
model_features = []     # feature columns used for training
time_forecast  = []     # Phase 3: time window risk list
peak_period    = ""     # Phase 3: name of peak crime period
peak_hour      = 0      # Phase 3: hour with most crimes

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Startup: Load persisted model so a server restart keeps the brain
if os.path.exists('model.pkl'):
    try:
        _saved = joblib.load('model.pkl')
        trained_model  = _saved.get('clf')
        label_encoder  = _saved.get('le')
        model_features = _saved.get('features', [])
        pca_model      = _saved.get('pca')           # restore fitted PCA
        _loaded_cfg    = _saved.get('col_config', {})
        if _loaded_cfg:
            col_config.update(_loaded_cfg)           # restore column mapping
        print("[Startup] Loaded persisted model from model.pkl")
    except Exception as _e:
        print(f"[Startup] Could not load model.pkl: {_e}")

# Startup: Reload the CSV so Map and Dashboard have data after a restart.
# Without this, crime_df stays None even though the model is loaded.
_csv_path = os.path.join('uploads', 'crime_dataset.csv')
if os.path.exists(_csv_path) and col_config.get('lat_col'):
    try:
        try:
            crime_df = pd.read_csv(_csv_path, encoding='utf-8',
                                   on_bad_lines='skip', engine='python')
        except UnicodeDecodeError:
            crime_df = pd.read_csv(_csv_path, encoding='latin-1',
                                   on_bad_lines='skip', engine='python')

        # Re-run DBSCAN so cluster column exists (needed by map + predict)
        crime_df = run_dbscan(crime_df,
                              col_config['lat_col'], col_config['lon_col'])

        # Rebuild cluster stats so /map has risk levels
        cluster_stats = compute_cluster_stats(crime_df, col_config['crime_col'])

        # Rebuild time forecast so /dashboard has the forecast panel
        from forecast import compute_time_forecast
        time_forecast, peak_period, peak_hour = compute_time_forecast(crime_df)

        print(f"[Startup] Reloaded {len(crime_df)} crime records from CSV")
    except Exception as _e:
        print(f"[Startup] Could not reload crime CSV: {_e}")
        crime_df = None


# ==============================
# HELPER: LOGIN REQUIRED
# ==============================
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash("Please login first.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ==============================
# PHASE 1: DBSCAN CLUSTERING
# ==============================
def run_dbscan(df, lat_col, lon_col, eps=None, min_samples=3):
    """
    Clusters crime locations using DBSCAN with Haversine distance.
    If eps is not provided, it is auto-detected using the K-Distance Graph
    elbow point method for optimal cluster detection.
    Falls back to eps=0.01 if auto-detection fails or data is too small.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    # Step 1: Convert lat/lon to radians (required for haversine metric)
    coords     = df[[lat_col, lon_col]].values
    coords_rad = np.radians(coords)

    # Step 2: Auto-detect eps using K-Distance Graph if not provided
    if eps is None:
        try:
            # Need at least min_samples + 1 points to compute neighbours
            if len(coords_rad) < min_samples + 1:
                raise ValueError("Too few data points for auto eps detection.")

            # Fit KNN with the same number of neighbours as min_samples
            nbrs = NearestNeighbors(
                n_neighbors  = min_samples,
                algorithm    = 'ball_tree',
                metric       = 'haversine'
            ).fit(coords_rad)

            # Get distance to the k-th nearest neighbour for every point
            distances, _ = nbrs.kneighbors(coords_rad)
            k_distances  = distances[:, -1]   # last column = k-th neighbour

            # Step 3: Sort distances in ascending order
            k_distances_sorted = np.sort(k_distances)

            # Step 4: Find the Elbow Point — point of maximum curvature
            # We compute the second derivative (acceleration) of the sorted
            # distance curve. The index with the highest value is the elbow.
            n      = len(k_distances_sorted)
            x_vals = np.arange(n)

            # First derivative (slope)
            first_deriv  = np.gradient(k_distances_sorted, x_vals)
            # Second derivative (curvature / acceleration)
            second_deriv = np.gradient(first_deriv, x_vals)

            # Elbow = index where curvature is maximum
            elbow_idx = int(np.argmax(second_deriv[5:]) + 5)
            eps       = float(k_distances_sorted[elbow_idx])

            # Clamp: keep eps in a safe usable range
            eps = max(0.001, min(eps, 0.5))

            print(f"[DBSCAN] Auto-detected eps = {eps:.6f} (elbow at index {elbow_idx})")

        except Exception as e:
            # Safety fallback — never crash the upload
            print(f"[DBSCAN] eps auto-detection failed ({e}). Using fallback eps=0.01")
            eps = 0.01

    # Step 5: Run DBSCAN with the chosen eps
    db            = DBSCAN(
        eps         = eps,
        min_samples = min_samples,
        algorithm   = 'ball_tree',
        metric      = 'haversine'
    )
    df            = df.copy()
    df['cluster'] = db.fit_predict(coords_rad)

    return df


def compute_cluster_stats(df, crime_col):
    """
    Classifies each cluster as high/medium/low based on crime count.
    Returns dict keyed by cluster ID.
    """
    stats = {}
    for cid in df['cluster'].unique():
        subset     = df[df['cluster'] == cid]
        count      = len(subset)
        top_crimes = subset[crime_col].value_counts().head(3).to_dict()
        if cid == -1:       risk = 'low'
        elif count >= 10:   risk = 'high'
        elif count >= 5:    risk = 'medium'
        else:               risk = 'low'
        stats[int(cid)] = {'count': count, 'risk': risk, 'top_crimes': top_crimes}
    return stats


# ==============================
# PHASE 1: RANDOM FOREST
# ==============================
def train_random_forest(df, lat_col, lon_col, crime_col):
    """
    Trains Random Forest on location + time features.
    Uses PCA if 4+ features available.
    FIT PCA here on training data only — saved to global pca_model.
    predict route must only call pca_model.transform(), never fit again.
    """
    global pca_model

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings('ignore')

    feature_cols = [lat_col, lon_col]
    if 'cluster' in df.columns:
        feature_cols.append('cluster')
    for col in ['hour', 'hour_of_day', 'day_of_week', 'day', 'month']:
        if col in df.columns:
            feature_cols.append(col)

    X  = df[feature_cols].copy().fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(df[crime_col].astype(str))

    if X.shape[1] >= 4:
        pca           = PCA(n_components=min(X.shape[1], 4))
        X_transformed = pca.fit_transform(X)
        pca_model     = pca          # save fitted PCA globally
    else:
        X_transformed = X.values
        pca_model     = None         # no PCA needed for small feature sets

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_transformed, y)
    return clf, le, feature_cols, X_transformed.shape[1]


# ==============================
# PHASE 1 + PHASE 3 + SOCIAL SIEVE: FOLIUM MAP
# ==============================
def generate_crime_map(df, lat_col, lon_col, crime_col, city_col,
                       social_posts=None, red_alert_clusters=None):
    """
    Generates Folium map with color-coded cluster markers.
    Phase 3: popup includes peak crime time per cluster.
    Social Sieve: Red Alert clusters highlighted; social pulse layer added.
    """
    import folium
    from forecast import cluster_time_forecast

    # Default to empty lists if Social Sieve data not provided
    if social_posts is None:
        social_posts = []
    if red_alert_clusters is None:
        red_alert_clusters = []

    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles='CartoDB dark_matter')

    risk_colors = {'high': '#e63946', 'medium': '#f4a261', 'low': '#2ec97a'}

    # --- Existing DBSCAN cluster markers ---
    for cid, info in cluster_stats.items():
        cluster_rows = df[df['cluster'] == cid]
        if len(cluster_rows) == 0:
            continue

        color  = risk_colors.get(info['risk'], '#4a9eff')
        radius = 8 if info['risk'] == 'low' else (14 if info['risk'] == 'medium' else 20)

        # Social Sieve: boost appearance for Red Alert clusters
        is_red_alert = (int(cid) in red_alert_clusters)
        if is_red_alert:
            color  = '#ff0022'   # brighter red
            radius = 28          # larger circle

        try:
            peak_time = cluster_time_forecast(df, cid)
        except:
            peak_time = "N/A"

        for _, row in cluster_rows.iterrows():
            top_crime = list(info['top_crimes'].keys())[0] if info['top_crimes'] else 'Unknown'
            label     = 'NOISE / ISOLATED' if cid == -1 else f'CLUSTER {cid}'

            # Social Sieve: add red alert banner inside popup if applicable
            red_alert_html = ""
            if is_red_alert:
                red_alert_html = """
                <div style="background:#ff0022;color:#fff;font-weight:bold;
                            padding:5px 8px;font-size:12px;margin-bottom:6px;
                            letter-spacing:0.05em;">
                    🚨 RED ALERT — Social Pulse Detected
                </div>"""

            popup_html = f"""
            <div style="font-family:monospace;background:#111;color:#e8e8f0;
                        padding:10px;min-width:190px;border-left:3px solid {color};">
                {red_alert_html}
                <div style="color:{color};font-weight:bold;font-size:13px;">{label}</div>
                <div style="color:#aaa;font-size:11px;margin-top:4px;">
                    Risk: <span style="color:{color};">{info['risk'].upper()}</span></div>
                <div style="color:#aaa;font-size:11px;">
                    Crimes: <b style="color:#fff;">{info['count']}</b></div>
                <div style="color:#aaa;font-size:11px;">
                    Top Crime: <b style="color:#fff;">{top_crime}</b></div>
                <div style="color:#aaa;font-size:11px;">City: {row.get(city_col, 'N/A')}</div>
                <div style="color:#aaa;font-size:11px;">Crime: {row.get(crime_col, 'N/A')}</div>
                <hr style="border-color:#333;margin:6px 0;">
                <div style="color:#f4a261;font-size:10px;">⏰ HIGH-RISK TIME</div>
                <div style="color:#fff;font-size:11px;">Peak: {peak_time}</div>
            </div>"""
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=radius, color=color, fill=True,
                fill_color=color, fill_opacity=0.45, weight=1.5,
                popup=folium.Popup(popup_html, max_width=260)
            ).add_to(m)

    # --- Social Sieve: add social pulse markers as a separate layer ---
    if social_posts:
        pulse_layer = folium.FeatureGroup(name="📡 Social Pulse (Live)")

        # Build city → (lat, lon) lookup from the dataset
        city_col_name = col_config.get('city_col', city_col)
        city_coords   = {}
        for _, row in df.iterrows():
            city_name = str(row.get(city_col_name, '')).strip().lower()
            if city_name and city_name not in city_coords:
                city_coords[city_name] = (float(row[lat_col]), float(row[lon_col]))

        pulse_colors = {'high': '#ff0022', 'medium': '#f4a261', 'low': '#00e5ff'}

        for post in social_posts:
            city_key = post['city'].strip().lower()
            coords   = city_coords.get(city_key)
            if coords is None:
                continue

            p_color = pulse_colors.get(post['severity'], '#ffffff')

            pulse_popup = f"""
            <div style="font-family:monospace;background:#0d0d18;color:#e8e8f0;
                        padding:10px;min-width:200px;border-left:3px solid {p_color};">
                <div style="color:{p_color};font-weight:bold;font-size:12px;">
                    📡 SOCIAL PULSE</div>
                <div style="color:#aaa;font-size:11px;margin-top:4px;">
                    City: <b style="color:#fff;">{post['city']}</b></div>
                <div style="color:#ccc;font-size:11px;margin-top:6px;font-style:italic;">
                    "{post['text']}"</div>
                <hr style="border-color:#333;margin:6px 0;">
                <div style="color:#aaa;font-size:10px;">
                    🔍 Keyword: <span style="color:{p_color};">{post['keyword']}</span></div>
                <div style="color:#aaa;font-size:10px;">
                    Severity: <span style="color:{p_color};text-transform:uppercase;">
                    {post['severity']}</span></div>
                <div style="color:#aaa;font-size:10px;">Source: {post['source']}</div>
                <div style="color:#aaa;font-size:10px;">Time: {post['time_str']}</div>
            </div>"""

            folium.CircleMarker(
                location=coords,
                radius=6,
                color=p_color,
                fill=True,
                fill_color=p_color,
                fill_opacity=0.7,
                weight=2,
                popup=folium.Popup(pulse_popup, max_width=260),
                tooltip=f"📡 {post['city']} — {post['severity'].upper()}"
            ).add_to(pulse_layer)

        pulse_layer.add_to(m)
        folium.LayerControl().add_to(m)

    m.save(os.path.join('static', 'crime_map_embed.html'))


# ==============================
# PHASE 4: CONTEXT ADJUSTMENT
# ==============================
def get_context_factors(hour, day):
    """
    Checks time and day context to identify risk-increasing conditions.
    Returns a list of human-readable context strings.
    """
    factors = []
    if hour >= 20 or hour <= 2:
        factors.append("🌙 Night time (8PM–2AM) — higher risk")
    if day in [6, 7]:
        factors.append("📅 Weekend — higher crime activity")
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        factors.append("🚗 Rush hour — elevated street crime")
    return factors


# ==============================
# PHASE 4: CRIME EXPLANATION ENGINE
# ==============================
def generate_explanation(crime_type, cluster_id, hour, confidence, cluster_stats):
    """
    Rule-based explanation for the prediction.
    Uses cluster density, time patterns, and confidence score.
    No external libraries — plain Python logic only.
    """
    reasons = []

    # Reason 1: Cluster density
    if cluster_id is not None and int(cluster_id) in cluster_stats:
        info  = cluster_stats[int(cluster_id)]
        risk  = info.get('risk', 'low')
        count = info.get('count', 0)
        if risk == 'high':
            reasons.append(f"📍 High crime density cluster ({count} crimes in area)")
        elif risk == 'medium':
            reasons.append(f"📍 Medium density cluster ({count} crimes in area)")
        else:
            reasons.append(f"📍 Low density area ({count} crimes recorded)")

    # Reason 2: Time pattern per crime type
    time_patterns = {
        'Theft'           : "common during daytime hours",
        'Burglary'        : "peaks at night when homes are empty",
        'Vehicle Theft'   : "most common late night / early morning",
        'Murder'          : "higher frequency after dark",
        'Robbery'         : "evening and night activity",
        'Assault on Women': "rises in evening hours",
        'Rape'            : "predominantly night time",
        'Road Accident'   : "throughout the day on busy roads",
        'Hit and Run'     : "spread across daylight hours",
        'Extortion'       : "during business hours",
        'Kidnapping'      : "daytime targeting common",
        'Dacoity'         : "late night gang activity",
        'Stalking'        : "evening through night hours",
        'Sexual Harassment': "busy public hours",
    }
    note = time_patterns.get(crime_type, "pattern detected in this area")
    reasons.append(f"⏰ {crime_type} — {note} (input hour: {hour}:00)")

    # Reason 3: Confidence interpretation
    if confidence >= 70:
        reasons.append(f"🎯 High model confidence ({confidence}%) — strong historical pattern match")
    elif confidence >= 45:
        reasons.append(f"🎯 Moderate confidence ({confidence}%) — likely pattern in this zone")
    else:
        reasons.append(f"🎯 Lower confidence ({confidence}%) — mixed signals, treat as indicative")

    return reasons


# ==============================
# PHASE 2: ANALYTICS HELPERS
# ==============================
def get_city_analysis(df, city_col, crime_col):
    """Phase 2: Returns top 10 cities by crime count."""
    return df[city_col].value_counts().head(10).to_dict()


def get_crime_type_breakdown(df, crime_col):
    """Phase 2: Returns crime type distribution as dict."""
    return df[crime_col].value_counts().to_dict()


def get_top_risky_cities(df, city_col, crime_col, top_n=5):
    """Phase 2: Returns top N cities with their most common crime."""
    result = []
    top_cities = df[city_col].value_counts().head(top_n).index.tolist()
    for city in top_cities:
        subset      = df[df[city_col] == city]
        count       = len(subset)
        top_crime   = subset[crime_col].value_counts().index[0] if len(subset) > 0 else 'N/A'
        result.append({'city': city, 'count': count, 'top_crime': top_crime})
    return result


def get_hourly_trend(df):
    """Phase 2: Returns crime count per hour for trend chart."""
    if 'hour' not in df.columns:
        return {}
    return df['hour'].value_counts().sort_index().to_dict()


# ==============================
# ROUTES
# ==============================

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username   = request.form.get('username', '').strip()
        password   = request.form.get('password', '').strip()
        secret_key = request.form.get('secret_key', '').strip()
        if not username or not password or not secret_key:
            flash("All fields are required.", "danger")
        elif secret_key != OFFICER_SECRET_KEY:
            flash("Invalid officer secret key.", "danger")
        else:
            if register_user(username, password):
                flash("Registration successful! Please login.", "success")
                return redirect(url_for('login'))
            else:
                flash("Username already exists.", "danger")
    return render_template("register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if login_user(username, password):
            session['username'] = username
            flash(f"Welcome {username}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route('/dashboard')
@login_required
def dashboard():
    """
    Phase 1: Core stats.
    Phase 3: Time forecast panel.
    Phase 4: Feedback count card.
    """
    global crime_df, time_forecast, peak_period, peak_hour

    if crime_df is not None:
        stats = {
            "dataset_loaded": True,
            "total_crimes"  : len(crime_df),
            "cities"        : crime_df[col_config.get('city_col', '')].nunique()
                              if col_config.get('city_col') in crime_df.columns else 0,
            "crime_types"   : crime_df[col_config.get('crime_col', '')].nunique()
                              if col_config.get('crime_col') in crime_df.columns else 0,
        }
    else:
        stats = {"dataset_loaded": False, "total_crimes": 0, "cities": 0, "crime_types": 0}

    return render_template("dashboard.html",
        username       = session['username'],
        stats          = stats,
        time_forecast  = time_forecast,
        peak_period    = peak_period,
        peak_hour      = peak_hour,
        feedback_count = get_feedback_count(),  # Phase 4
    )


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """
    Phase 1: Upload CSV, run DBSCAN, train RF.
    Phase 3: Compute time forecast after upload.
    """
    global crime_df, col_config, cluster_stats
    global trained_model, label_encoder, model_features, pca_model
    global time_forecast, peak_period, peak_hour

    if request.method == 'POST':
        file      = request.files.get('file')
        lat_col   = request.form.get('lat_col')
        lon_col   = request.form.get('lon_col')
        city_col  = request.form.get('city_col')
        crime_col = request.form.get('crime_col')

        if not file or file.filename == '':
            flash("No file selected.", "danger")
            return redirect(url_for('upload'))
        if not file.filename.endswith('.csv'):
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for('upload'))
        if not all([lat_col, lon_col, city_col, crime_col]):
            flash("Please select all four column dropdowns.", "danger")
            return redirect(url_for('upload'))

        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'crime_dataset.csv')
            file.save(filepath)

            try:
                df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', engine='python')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', engine='python')

            missing = [c for c in [lat_col, lon_col, city_col, crime_col] if c not in df.columns]
            if missing:
                flash(f"Columns not found: {', '.join(missing)}", "danger")
                return redirect(url_for('upload'))

            df = df.dropna(subset=[lat_col, lon_col, city_col, crime_col])
            df = run_dbscan(df, lat_col, lon_col)
            cluster_stats = compute_cluster_stats(df, crime_col)

            clf, le, feat_cols, n_pca = train_random_forest(df, lat_col, lon_col, crime_col)
            trained_model  = clf
            label_encoder  = le
            model_features = feat_cols
            crime_df       = df
            col_config     = {
                'lat_col'  : lat_col, 'lon_col': lon_col,
                'city_col' : city_col, 'crime_col': crime_col, 'n_pca': n_pca
            }

            # Fix 5: Persist model so a server restart doesn't lose the brain
            joblib.dump(
                {'clf': clf, 'le': le, 'features': feat_cols, 'n_pca': n_pca,
                 'pca': pca_model, 'col_config': col_config},
                'model.pkl'
            )

            from forecast import compute_time_forecast
            time_forecast, peak_period, peak_hour = compute_time_forecast(df)

            n_clusters = len([c for c in df['cluster'].unique() if c != -1])
            flash(f"Dataset uploaded! {len(df)} records, {n_clusters} clusters. Model trained.", "success")
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "danger")
            return redirect(url_for('upload'))

    dataset_loaded = crime_df is not None
    total_crimes   = len(crime_df) if crime_df is not None else 0
    cities = (crime_df[col_config.get('city_col', '')].nunique()
              if crime_df is not None and col_config.get('city_col') in crime_df.columns else 0)

    return render_template("upload.html",
        username=session['username'],
        dataset_loaded=dataset_loaded,
        total_crimes=total_crimes,
        cities=cities
    )


@app.route('/map')
@login_required
def crime_map():
    """
    Phase 1 + Phase 3: Folium map with cluster markers and peak time popups.
    Social Sieve: social posts and red alert clusters added to map and sidebar.
    """
    if crime_df is None:
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for('upload'))

    # --- Social Sieve: generate posts and find red alert clusters ---
    # Safe import — map still works perfectly if social_sieve.py is missing
    try:
        from social_sieve import generate_mock_social_posts, check_high_risk_pulse

        # Get unique city names from the uploaded dataset
        city_list    = crime_df[col_config['city_col']].dropna().unique().tolist()

        # Generate 12 mock social media posts linked to dataset cities
        social_posts = generate_mock_social_posts(city_list, n_posts=12)

        # Cross-reference posts with DBSCAN clusters to find Red Alerts
        red_alert_clusters = check_high_risk_pulse(
            social_posts, cluster_stats, crime_df,
            col_config['lat_col'], col_config['lon_col'],
            trained_model, label_encoder, model_features, col_config
        )

    except Exception:
        # social_sieve.py missing or raised a runtime error — skip silently
        social_posts       = []
        red_alert_clusters = []

    # Generate the Folium map (with Social Sieve data)
    try:
        generate_crime_map(
            crime_df,
            col_config['lat_col'], col_config['lon_col'],
            col_config['crime_col'], col_config['city_col'],
            social_posts=social_posts,
            red_alert_clusters=red_alert_clusters
        )
    except Exception as e:
        flash(f"Error generating map: {str(e)}", "danger")
        return redirect(url_for('dashboard'))

    high_count   = len([c for c, s in cluster_stats.items() if s['risk'] == 'high'   and c != -1])
    medium_count = len([c for c, s in cluster_stats.items() if s['risk'] == 'medium' and c != -1])
    low_count    = len([c for c, s in cluster_stats.items() if s['risk'] == 'low'    and c != -1])

    map_summary = {
        'total_records'  : len(crime_df),
        'total_clusters' : high_count + medium_count + low_count,
        'high_risk'      : high_count,
        'medium_risk'    : medium_count,
        'low_risk'       : low_count,
        'noise_points'   : cluster_stats.get(-1, {}).get('count', 0),
        # Social Sieve: red alert count for sidebar stat card
        'red_alert_count': len(red_alert_clusters),
    }

    top_crimes   = crime_df[col_config['crime_col']].value_counts().head(5).to_dict()
    risky        = {k: v for k, v in cluster_stats.items() if k != -1}
    top_clusters = sorted(risky.items(), key=lambda x: x[1]['count'], reverse=True)[:5]

    return render_template("map.html",
        username           = session['username'],
        map_summary        = map_summary,
        top_crimes         = top_crimes,
        top_clusters       = top_clusters,
        # Social Sieve extras passed to template
        social_posts       = social_posts,
        red_alert_clusters = red_alert_clusters,
    )


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """
    Phase 1: Random Forest crime type prediction.
    Phase 3: Crime probability score.
    Phase 4: Context adjustment + explanation engine.
    """
    if crime_df is None or trained_model is None:
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for('upload'))

    prediction_result  = None
    probability_result = None

    if request.method == 'POST':
        try:
            from forecast import get_crime_probability

            lat  = float(request.form.get('latitude'))
            lon  = float(request.form.get('longitude'))
            hour = float(request.form.get('hour', 12))
            day  = float(request.form.get('day_of_week', 1))

            # Phase 4: Context factors
            context_factors = get_context_factors(int(hour), int(day))

            row = {
                col_config['lat_col']: lat, col_config['lon_col']: lon,
                'cluster': 0, 'hour': hour, 'hour_of_day': hour,
                'day_of_week': day, 'day': day, 'month': 1
            }
            input_vals = [row.get(f, 0) for f in model_features]
            X_input    = np.array(input_vals).reshape(1, -1)

            if pca_model is not None and len(model_features) >= 4 and X_input.shape[1] >= 4:
                # Fix: never fit PCA on predict data — use the model fitted during training
                X_input = pca_model.transform(X_input)

            # Phase 1: prediction
            proba      = trained_model.predict_proba(X_input)[0]
            pred_idx   = np.argmax(proba)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = round(float(proba[pred_idx]) * 100, 1)
            top3_idx   = np.argsort(proba)[::-1][:3]
            top3       = [(label_encoder.inverse_transform([i])[0],
                           round(float(proba[i]) * 100, 1)) for i in top3_idx]

            # Phase 4: Find nearest cluster for explanation
            nearest_cluster = None
            lat_col_n = col_config['lat_col']
            lon_col_n = col_config['lon_col']
            for cid, info in cluster_stats.items():
                if cid == -1:
                    continue
                c_rows = crime_df[crime_df['cluster'] == cid]
                if len(c_rows) == 0:
                    continue
                dist = abs(c_rows[lat_col_n].mean() - lat) + abs(c_rows[lon_col_n].mean() - lon)
                if nearest_cluster is None or dist < nearest_cluster[1]:
                    nearest_cluster = (cid, dist)

            nearest_cid = nearest_cluster[0] if nearest_cluster else None
            explanation = generate_explanation(pred_label, nearest_cid, int(hour),
                                               confidence, cluster_stats)

            prediction_result = {
                'crime_type'     : pred_label,
                'confidence'     : confidence,
                'top3'           : top3,
                'lat'            : lat,
                'lon'            : lon,
                'hour'           : int(hour),
                'day'            : int(day),
                'explanation'    : explanation,      # Phase 4
                'context_factors': context_factors,  # Phase 4
            }

            # Phase 3: probability score
            probability_result = get_crime_probability(
                trained_model, label_encoder, model_features,
                col_config, crime_df, lat, lon, hour, day
            )

        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")

    crime_types    = crime_df[col_config['crime_col']].value_counts().index.tolist()
    feedback_count = get_feedback_count()  # Phase 4

    return render_template("predict.html",
        username       = session['username'],
        prediction     = prediction_result,
        probability    = probability_result,
        crime_types    = crime_types,
        feedback_count = feedback_count,
    )


# ==============================
# PHASE 2: ANALYTICS ROUTE
# ==============================
@app.route('/analytics')
@login_required
def analytics():
    """
    Phase 2: City analysis, crime type breakdown,
    top risky cities, hourly trend.
    """
    if crime_df is None:
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for('upload'))

    city_col  = col_config['city_col']
    crime_col = col_config['crime_col']

    city_analysis     = get_city_analysis(crime_df, city_col, crime_col)
    crime_breakdown   = get_crime_type_breakdown(crime_df, crime_col)
    top_risky_cities  = get_top_risky_cities(crime_df, city_col, crime_col)
    hourly_trend      = get_hourly_trend(crime_df)

    # Top crime type overall
    top_crime_type = crime_df[crime_col].value_counts().index[0] if len(crime_df) > 0 else 'N/A'

    # City with most crimes
    top_city = crime_df[city_col].value_counts().index[0] if len(crime_df) > 0 else 'N/A'

    # Crime type filter support — filter by selected crime type
    selected_crime = request.args.get('filter_crime', '')
    filtered_df    = crime_df
    if selected_crime and selected_crime in crime_df[crime_col].values:
        filtered_df = crime_df[crime_df[crime_col] == selected_crime]

    # Data table — top 50 rows of filtered data
    table_data = filtered_df[[city_col, crime_col] +
                  (['hour'] if 'hour' in filtered_df.columns else []) +
                  [col_config['lat_col'], col_config['lon_col']]
                 ].head(50).to_dict('records')

    return render_template("analytics.html",
        username        = session['username'],
        city_analysis   = city_analysis,
        crime_breakdown = crime_breakdown,
        top_risky_cities= top_risky_cities,
        hourly_trend    = hourly_trend,
        top_crime_type  = top_crime_type,
        top_city        = top_city,
        table_data      = table_data,
        crime_types     = crime_df[crime_col].unique().tolist(),
        selected_crime  = selected_crime,
        total_records   = len(crime_df),
    )


# ==============================
# PHASE 4: SAVE FEEDBACK ROUTE
# ==============================
@app.route('/save_feedback', methods=['POST'])
@login_required
def save_feedback_route():
    """
    Phase 4: Receives officer feedback on a prediction.
    Stores predicted vs actual crime type in the feedback table.
    """
    predicted_crime = request.form.get('predicted_crime', '').strip()
    actual_crime    = request.form.get('actual_crime', '').strip()
    latitude        = request.form.get('latitude', 0)
    longitude       = request.form.get('longitude', 0)
    was_correct     = request.form.get('was_correct', 'no')

    # If officer confirmed correct, actual = predicted
    if was_correct == 'yes':
        actual_crime = predicted_crime

    try:
        lat = float(latitude)
        lon = float(longitude)
    except:
        lat, lon = 0.0, 0.0

    ok = save_feedback(predicted_crime, actual_crime, lat, lon)
    if ok:
        flash("✔ Feedback saved successfully. Thank you!", "success")
    else:
        flash("Could not save feedback.", "danger")

    return redirect(url_for('predict'))


@app.route('/retrain_model', methods=['POST'])
@login_required
def retrain_model():
    """
    Phase 4 — Feedback Loop: Retrain Random Forest by merging officer
    feedback records with the original crime_df.
    The AI learns from officer corrections.
    """
    global trained_model, label_encoder, model_features, pca_model, crime_df, col_config, cluster_stats

    if crime_df is None:
        flash("No dataset loaded. Please upload a dataset first.", "warning")
        return redirect(url_for('upload'))

    try:
        from db import get_all_feedback   # fetch every feedback row

        feedback_rows = get_all_feedback()

        if not feedback_rows:
            flash("No feedback records found. Train officers to submit feedback first.", "warning")
            return redirect(url_for('predict'))

        lat_col   = col_config['lat_col']
        lon_col   = col_config['lon_col']
        crime_col = col_config['crime_col']

        # Build a DataFrame from feedback records
        fb_df = pd.DataFrame(feedback_rows,
                             columns=['id', 'predicted_crime', 'actual_crime',
                                      'latitude', 'longitude', 'timestamp'])

        fb_df = fb_df.rename(columns={
            'latitude'     : lat_col,
            'longitude'    : lon_col,
            'actual_crime' : crime_col,
        })

        # Only keep rows where officer supplied an actual crime label
        fb_df = fb_df[fb_df[crime_col].notna() & (fb_df[crime_col] != '')]

        if fb_df.empty:
            flash("Feedback exists but no usable actual-crime labels found.", "warning")
            return redirect(url_for('predict'))

        # Always start from a FRESH copy of the original CSV — never from
        # crime_df which may already contain previous feedback rows.
        # This prevents data duplication when Retrain is clicked multiple times.
        _csv_path = os.path.join('uploads', 'crime_dataset.csv')
        if not os.path.exists(_csv_path):
            flash("Original CSV not found in uploads/. Please re-upload the dataset.", "danger")
            return redirect(url_for('upload'))

        try:
            base_df = pd.read_csv(_csv_path, encoding='utf-8',
                                  on_bad_lines='skip', engine='python')
        except UnicodeDecodeError:
            base_df = pd.read_csv(_csv_path, encoding='latin-1',
                                  on_bad_lines='skip', engine='python')

        base_df = base_df.dropna(subset=[lat_col, lon_col, col_config['city_col'], crime_col])

        # Copy only columns that exist in base_df so merge is clean
        common_cols = [c for c in base_df.columns if c in fb_df.columns]
        merged_df   = pd.concat([base_df, fb_df[common_cols]], ignore_index=True)

        # Re-run DBSCAN clusters on merged data and retrain model
        merged_df     = run_dbscan(merged_df, lat_col, lon_col)
        cluster_stats = compute_cluster_stats(merged_df, crime_col)   # keep map in sync
        clf, le, feat_cols, n_pca = train_random_forest(merged_df, lat_col, lon_col, crime_col)

        trained_model  = clf
        label_encoder  = le
        model_features = feat_cols
        crime_df       = merged_df
        col_config['n_pca'] = n_pca

        # Persist updated model to disk
        joblib.dump(
            {'clf': clf, 'le': le, 'features': feat_cols, 'n_pca': n_pca,
             'pca': pca_model, 'col_config': col_config},
            'model.pkl'
        )

        flash(f"✔ Model retrained on {len(merged_df)} records "
              f"({len(fb_df)} feedback rows merged). Brain updated!", "success")

    except ImportError:
        flash("get_all_feedback not found in db.py. Add it to enable retraining.", "danger")
    except Exception as e:
        flash(f"Retraining error: {str(e)}", "danger")

    return redirect(url_for('predict'))


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))


# ==============================
# START / INITIALIZATION
# ==============================
with app.app_context():
    create_tables()
    create_feedback_table()   # Phase 4
    setup_default_admin()
    setup_default_users()

if __name__ == "__main__":
    print("Server started at http://127.0.0.1:5000")
    print("Officer secret key: police@2025")
    app.run(debug=True)