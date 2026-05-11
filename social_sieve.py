# social_sieve.py
# Social Sieve — Crime Spot Detection Add-On
# Team 05 | Amrita Vishwa Vidyapeetham | 22MAT122
#
# This module simulates a social media intelligence layer.
# It generates mock social posts from cities in the dataset,
# detects crime keywords, assigns severity, and cross-references
# with DBSCAN cluster data to flag Red Alert clusters.

import random
from datetime import datetime, timedelta


# =====================================================
# POST TEMPLATES — realistic social media style texts
# =====================================================

# Each template has a keyword and severity embedded
POST_TEMPLATES = [
    {"text": "Heard loud gunshots near {city} market area, people running!",          "keyword": "gunshots",   "severity": "high"},
    {"text": "Robbery reported at {city} bus stand. Victim was beaten.",               "keyword": "robbery",    "severity": "high"},
    {"text": "Suspicious group seen with weapons near {city} railway station.",        "keyword": "weapons",    "severity": "high"},
    {"text": "A woman was harassed near {city} college road tonight.",                 "keyword": "harassment", "severity": "medium"},
    {"text": "Chain snatching incident reported in {city} main bazaar.",               "keyword": "snatching",  "severity": "medium"},
    {"text": "Someone broke into a car in {city} parking area. Be careful!",           "keyword": "break-in",   "severity": "medium"},
    {"text": "Street fight spotted near {city} signal. Police called.",               "keyword": "fight",      "severity": "medium"},
    {"text": "Drunk driving near {city} highway, biker almost hit pedestrian.",        "keyword": "drunk",      "severity": "low"},
    {"text": "Pickpocket active in {city} crowded market. Keep your bags close.",      "keyword": "pickpocket", "severity": "low"},
    {"text": "Stray dogs attacking people near {city} park at night.",                 "keyword": "attack",     "severity": "low"},
    {"text": "Suspicious person lurking near {city} school gate in the evening.",      "keyword": "suspicious", "severity": "medium"},
    {"text": "Fire broke out near {city} warehouse. Arson suspected.",                 "keyword": "arson",      "severity": "high"},
    {"text": "Multiple thefts reported in {city} residential area this week.",         "keyword": "theft",      "severity": "medium"},
    {"text": "Gang activity suspected near {city} abandoned building.",                "keyword": "gang",       "severity": "high"},
    {"text": "Eve teasing incident at {city} metro station. Police alerted.",          "keyword": "eve teasing","severity": "medium"},
]

# Social media source names (fake but realistic)
SOURCES = [
    "Twitter / @LocalAlert",
    "Facebook / CrimeWatch Group",
    "WhatsApp / Neighborhood Watch",
    "Telegram / SafeCity Bot",
    "Reddit / r/CityAlerts",
    "Instagram / @SafetyFirst",
]


# =====================================================
# FUNCTION 1 — Generate Mock Social Posts
# =====================================================

def generate_mock_social_posts(city_list, n_posts=10):
    """
    Generates a list of simulated social media crime posts.

    Each post is linked to a city from the actual uploaded dataset,
    so the social feed feels grounded in real data.

    Parameters:
        city_list : list of city names from the crime DataFrame
        n_posts   : how many posts to generate (default: 10)

    Returns:
        A list of dicts, each with:
        - city, text, keyword, severity, source, time_str
    """

    if not city_list:
        return []

    posts = []

    # We generate n_posts posts, cycling through templates randomly
    for i in range(n_posts):
        # Pick a random city from the dataset
        city = random.choice(city_list)

        # Pick a random post template
        template = random.choice(POST_TEMPLATES)

        # Fill in the city name into the post text
        text = template["text"].format(city=city)

        # Pick a random source platform
        source = random.choice(SOURCES)

        # Generate a fake time in the last 6 hours
        minutes_ago = random.randint(5, 360)
        post_time   = datetime.now() - timedelta(minutes=minutes_ago)
        time_str    = post_time.strftime("%I:%M %p")   # e.g. "09:45 PM"

        posts.append({
            "city"    : city,
            "text"    : text,
            "keyword" : template["keyword"],
            "severity": template["severity"],
            "source"  : source,
            "time_str": time_str,
        })

    # Sort: high severity posts first so officers see critical items at top
    severity_order = {"high": 0, "medium": 1, "low": 2}
    posts.sort(key=lambda p: severity_order.get(p["severity"], 3))

    return posts


# =====================================================
# FUNCTION 2 — Identify Red Alert Clusters
# =====================================================

def check_high_risk_pulse(social_posts, cluster_stats, crime_df,
                           lat_col, lon_col,
                           trained_model, label_encoder, model_features, col_config):
    """
    Cross-references social posts with DBSCAN cluster data.

    Logic:
    - Find cities that appear in high-severity social posts.
    - Check which DBSCAN clusters have records in those cities.
    - If a cluster is already 'high' risk AND it has matching social posts
      → mark it as a Red Alert cluster.

    Parameters:
        social_posts   : list of posts from generate_mock_social_posts()
        cluster_stats  : dict of cluster info from compute_cluster_stats()
        crime_df       : the main crime DataFrame
        lat_col        : latitude column name
        lon_col        : longitude column name
        trained_model  : Random Forest model (not used here, for future expansion)
        label_encoder  : label encoder (for future use)
        model_features : model feature list (for future use)
        col_config     : dict with city_col, crime_col, etc.

    Returns:
        A list of cluster IDs (ints) that are Red Alerts.
        e.g. [0, 3, 7]
    """

    red_alert_clusters = []

    if not social_posts or crime_df is None or cluster_stats is None:
        return red_alert_clusters

    city_col = col_config.get('city_col', '')
    if not city_col or city_col not in crime_df.columns:
        return red_alert_clusters

    # Step 1: Collect cities that have HIGH severity posts
    # (only high-severity triggers Red Alert — medium/low are just informational)
    high_severity_cities = set()
    for post in social_posts:
        if post.get("severity") == "high":
            high_severity_cities.add(post["city"].strip().lower())

    if not high_severity_cities:
        return red_alert_clusters

    # Step 2: For each cluster, check if any of its crime records
    # are from a city that matches a high-severity social post
    for cid, info in cluster_stats.items():
        # Skip noise cluster (DBSCAN label = -1)
        if cid == -1:
            continue

        # Only consider clusters already flagged as 'high' risk by DBSCAN
        if info.get('risk') != 'high':
            continue

        # Get all rows belonging to this cluster
        cluster_rows = crime_df[crime_df['cluster'] == cid]

        if len(cluster_rows) == 0:
            continue

        # Check if any city in this cluster matches our social post cities
        cluster_cities = cluster_rows[city_col].str.strip().str.lower().unique()

        for city in cluster_cities:
            if city in high_severity_cities:
                # Match found! This cluster has both high DBSCAN risk
                # AND active social media pulse → RED ALERT
                red_alert_clusters.append(int(cid))
                break   # No need to check more cities for this cluster

    return red_alert_clusters
