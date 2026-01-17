"""
SQLite database setup for SmartLift.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "smartlift.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exercise TEXT NOT NULL,
            bodyweight REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workout_id INTEGER NOT NULL,
            set_number INTEGER NOT NULL,
            weight REAL NOT NULL,
            reps INTEGER NOT NULL,
            rir INTEGER NOT NULL,
            FOREIGN KEY (workout_id) REFERENCES workouts(id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workout_id INTEGER NOT NULL,
            exercise TEXT NOT NULL,
            predicted_reps INTEGER,
            actual_reps INTEGER,
            weight REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workout_id) REFERENCES workouts(id)
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized → {DB_PATH}")


if __name__ == "__main__":
    init_db()
