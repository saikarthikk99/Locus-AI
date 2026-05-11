# forecast.py
# Phase 3 — Crime Spot Detection
# Team 05 | Amrita Vishwa Vidyapeetham | 22MAT122
#
# This module contains two new ML features:
#   Feature 1: Crime Probability Prediction
#   Feature 2: Crime Time Forecast
#
# Both functions take existing data structures (DataFrame, trained RF model)
# so no new data pipelines are needed — they plug into the Phase 1 system.

import pandas as pd
import numpy as np


# =============================================================
# FEATURE 1 — CRIME PROBABILITY PREDICTION
# =============================================================

def get_crime_probability(trained_model, label_encoder, model_features,
                          col_config, crime_df, lat, lon, hour, day):
    """
    Calculates the probability of crime occurring at a given location and time.

    How it works:
    - We feed the user's location + time into the existing Random Forest model.
    - predict_proba() returns a probability for EACH crime type.
    - The highest single probability = how "likely" that crime scenario is.
    - We scale this into a 0–100% crime risk score.

    Parameters:
        trained_model  : the Random Forest model trained during CSV upload
        label_encoder  : encodes/decodes crime type labels
        model_features : list of column names used as features
        col_config     : dict with lat_col, lon_col, crime_col, etc.
        crime_df       : the loaded crime DataFrame
        lat, lon       : user-entered coordinates
        hour           : hour of day (0–23)
        day            : day of week (1–7)

    Returns a dict with probability info, or None if model not ready.
    """
    from sklearn.decomposition import PCA

    if trained_model is None or label_encoder is None:
        return None

    try:
        # Step 1: Build the input row matching model_features
        # (same structure as regular prediction)
        row = {
            col_config['lat_col']: lat,
            col_config['lon_col']: lon,
            'cluster'      : 0,
            'hour'         : hour,
            'hour_of_day'  : hour,
            'day_of_week'  : day,
            'day'          : day,
            'month'        : 1
        }
        input_vals = [row.get(f, 0) for f in model_features]
        X_input    = np.array(input_vals).reshape(1, -1)

        # Step 2: Apply PCA if the model was trained with it
        n_pca = col_config.get('n_pca', len(model_features))
        if len(model_features) >= 4 and X_input.shape[1] >= 4:
            pca = PCA(n_components=n_pca)
            pca.fit(crime_df[model_features].fillna(0).values)
            X_input = pca.transform(X_input)

        # Step 3: Get probability scores for all crime types
        proba = trained_model.predict_proba(X_input)[0]

        # Step 4: The maximum probability = confidence in the top prediction
        # We use this as the "crime probability" at this location/time
        max_prob   = float(np.max(proba))
        prob_pct   = round(max_prob * 100, 1)

        # Step 5: Determine risk label based on probability percentage
        if prob_pct >= 70:
            risk_level = 'CRITICAL'
            risk_color = '#e63946'   # red
        elif prob_pct >= 50:
            risk_level = 'HIGH'
            risk_color = '#f4a261'   # orange
        elif prob_pct >= 30:
            risk_level = 'MODERATE'
            risk_color = '#4a9eff'   # blue
        else:
            risk_level = 'LOW'
            risk_color = '#2ec97a'   # green

        # Step 6: Get the top predicted crime type
        pred_idx    = int(np.argmax(proba))
        top_crime   = label_encoder.inverse_transform([pred_idx])[0]

        # Step 7: Build top 3 crime types with probabilities
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            (label_encoder.inverse_transform([i])[0], round(float(proba[i]) * 100, 1))
            for i in top3_idx
        ]

        return {
            'probability' : prob_pct,
            'risk_level'  : risk_level,
            'risk_color'  : risk_color,
            'top_crime'   : top_crime,
            'top3'        : top3,
            'lat'         : lat,
            'lon'         : lon,
            'hour'        : int(hour),
            'day'         : int(day),
        }

    except Exception as e:
        print(f"Probability calculation error: {e}")
        return None


# =============================================================
# FEATURE 2 — CRIME TIME FORECAST
# =============================================================

def compute_time_forecast(df):
    """
    Analyzes the dataset to find which hours of the day have the most crime.
    Groups crimes into readable time windows and assigns a risk level to each.

    How it works:
    - Count how many crimes happen each hour (0–23).
    - Group hours into 5 named time windows.
    - Compare each window's count against the dataset average.
    - Classify as Very High / High / Medium / Low.

    Returns a list of dicts, one per time window, sorted by crime count.
    Also returns the single peak hour and its period name.
    """

    # Need 'hour' column — if not present, return empty result
    if 'hour' not in df.columns:
        return [], "Unknown", 0

    # Step 1: Count crimes per hour of day
    hourly_counts = df['hour'].value_counts().sort_index()

    # Step 2: Define 5 time windows
    time_windows = [
        {'name': 'Late Night',  'label': '12AM – 5AM',  'hours': list(range(0, 6)),   'icon': '🌙'},
        {'name': 'Morning',     'label': '6AM – 11AM',  'hours': list(range(6, 12)),  'icon': '🌅'},
        {'name': 'Afternoon',   'label': '12PM – 5PM',  'hours': list(range(12, 18)), 'icon': '☀️'},
        {'name': 'Evening',     'label': '6PM – 8PM',   'hours': list(range(18, 21)), 'icon': '🌆'},
        {'name': 'Night',       'label': '9PM – 11PM',  'hours': list(range(21, 24)), 'icon': '🌃'},
    ]

    # Step 3: For each window, sum up crime counts
    window_results = []
    for window in time_windows:
        # Add up crimes for each hour in this window
        total = sum(hourly_counts.get(h, 0) for h in window['hours'])
        window_results.append({
            'name'  : window['name'],
            'label' : window['label'],
            'icon'  : window['icon'],
            'count' : int(total),
            'hours' : window['hours'],
        })

    # Step 4: Find average count to classify risk levels
    counts_only = [w['count'] for w in window_results]
    avg_count   = np.mean(counts_only) if counts_only else 1
    max_count   = max(counts_only)     if counts_only else 1

    # Step 5: Assign risk level to each window
    for w in window_results:
        ratio = w['count'] / avg_count if avg_count > 0 else 0

        if ratio >= 1.5:
            w['risk']       = 'Very High'
            w['risk_color'] = '#e63946'
            w['bar_pct']    = min(100, int((w['count'] / max_count) * 100))
        elif ratio >= 1.0:
            w['risk']       = 'High'
            w['risk_color'] = '#f4a261'
            w['bar_pct']    = min(100, int((w['count'] / max_count) * 100))
        elif ratio >= 0.6:
            w['risk']       = 'Medium'
            w['risk_color'] = '#4a9eff'
            w['bar_pct']    = min(100, int((w['count'] / max_count) * 100))
        else:
            w['risk']       = 'Low'
            w['risk_color'] = '#2ec97a'
            w['bar_pct']    = max(10, int((w['count'] / max_count) * 100))

    # Step 6: Sort by count descending (highest risk window first)
    window_results.sort(key=lambda x: x['count'], reverse=True)

    # Step 7: Find the single peak hour
    if len(hourly_counts) > 0:
        peak_hour = int(hourly_counts.idxmax())
        peak_count = int(hourly_counts.max())
    else:
        peak_hour  = 0
        peak_count = 0

    # Map peak hour to a period name
    if 0 <= peak_hour <= 5:
        peak_period = "Late Night"
    elif 6 <= peak_hour <= 11:
        peak_period = "Morning"
    elif 12 <= peak_hour <= 17:
        peak_period = "Afternoon"
    elif 18 <= peak_hour <= 20:
        peak_period = "Evening"
    else:
        peak_period = "Night"

    return window_results, peak_period, peak_hour


# =============================================================
# HELPER — TIME FORECAST FOR INDIVIDUAL CLUSTERS
# =============================================================

def cluster_time_forecast(df, cluster_id):
    """
    Computes the peak crime time for a single cluster.
    Used to add time info to Folium map popups.

    Returns a short string like "Peak: 10PM (Night)"
    """
    if 'hour' not in df.columns or 'cluster' not in df.columns:
        return "N/A"

    # Filter to this cluster only
    subset = df[df['cluster'] == cluster_id]

    if len(subset) == 0:
        return "N/A"

    # Find most common hour in this cluster
    peak_hour = int(subset['hour'].mode().iloc[0])

    # Convert to 12-hour format for readability
    if peak_hour == 0:
        hour_str = "12AM"
    elif peak_hour < 12:
        hour_str = f"{peak_hour}AM"
    elif peak_hour == 12:
        hour_str = "12PM"
    else:
        hour_str = f"{peak_hour - 12}PM"

    # Get the period name
    if 0 <= peak_hour <= 5:
        period = "Late Night"
    elif 6 <= peak_hour <= 11:
        period = "Morning"
    elif 12 <= peak_hour <= 17:
        period = "Afternoon"
    elif 18 <= peak_hour <= 20:
        period = "Evening"
    else:
        period = "Night"

    return f"{hour_str} ({period})"
