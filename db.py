# db.py
# Crime Spot Detection — All Phases Combined
# Phase 1: User auth (users table)
# Phase 4: Prediction feedback (feedback table)

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE = 'crime_data.db'


def get_db_connection():
    """Opens a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------------------------
# PHASE 1 — USERS TABLE
# -----------------------------------------------

def create_tables():
    """Creates the users table if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def register_user(username, password):
    """Registers a new officer with hashed password."""
    hashed = generate_password_hash(password)
    conn   = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, hashed)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    """Returns True if credentials are correct."""
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE username = ?', (username,)
    ).fetchone()
    conn.close()
    if user and check_password_hash(user['password'], password):
        return True
    return False


def setup_default_admin():
    """Creates admin account on first run."""
    conn  = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    conn.close()
    if count == 0:
        register_user('admin', 'admin123')
        print("Default admin: admin / admin123")


def setup_default_users():
    """Creates default team accounts with password 'crime'."""
    for username in ['sats', 'nikesh', 'rahul', 'karthik']:
        ok = register_user(username, 'crime')
        if ok:
            print(f"Default user: {username} / crime")
        else:
            print(f"User '{username}' already exists.")


# -----------------------------------------------
# PHASE 4 — FEEDBACK TABLE
# -----------------------------------------------

def create_feedback_table():
    """
    Creates the feedback table for Phase 4 Model Feedback Learning.
    Stores officer corrections to predictions.
    """
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_crime TEXT    NOT NULL,
            actual_crime    TEXT,
            latitude        REAL,
            longitude       REAL,
            timestamp       TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    ''')
    conn.commit()
    conn.close()


def save_feedback(predicted_crime, actual_crime, latitude, longitude):
    """Saves one prediction feedback record."""
    conn = get_db_connection()
    try:
        conn.execute(
            '''INSERT INTO feedback (predicted_crime, actual_crime, latitude, longitude)
               VALUES (?, ?, ?, ?)''',
            (predicted_crime, actual_crime, latitude, longitude)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Feedback save error: {e}")
        return False
    finally:
        conn.close()


def get_feedback_count():
    """Returns total number of feedback entries collected."""
    conn  = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
    conn.close()
    return count


def get_all_feedback():
    """
    Returns all feedback rows as a list of tuples for retraining.
    Columns: (id, predicted_crime, actual_crime, latitude, longitude, timestamp)
    Used by /retrain_model in app.py to merge officer corrections with crime_df.
    """
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT id, predicted_crime, actual_crime, latitude, longitude, timestamp FROM feedback'
    ).fetchall()
    conn.close()
    return [tuple(row) for row in rows]