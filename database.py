"""
Database initialization and connection management for the Healthcare Recommendation System.
"""
import sqlite3
from typing import Dict, List, Optional, Any

import os
import config as config_module

def get_db_path():
    """Get database path, creating data directory if needed."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, config_module.DB_NAME)


def get_connection(db_name: str = None) -> sqlite3.Connection:
    """
    Create a database connection with row factory enabled.
    """
    if db_name is None:
        db_name = get_db_path()
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Drop existing tables and create fresh patient and provider tables.
    """
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS patients")
    cur.execute("DROP TABLE IF EXISTS providers")

    cur.execute("""
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            age INTEGER,
            sex TEXT,
            condition_tags TEXT,
            risk_score REAL,
            urgency_score REAL
        )
    """)

    cur.execute("""
        CREATE TABLE providers (
            provider_id TEXT PRIMARY KEY,
            speciality TEXT,
            conditions_treated TEXT,
            expertise_score REAL,
            success_rate REAL,
            availability_score REAL,
            load_score REAL
        )
    """)

    conn.commit()


def seed_sample_data(conn: sqlite3.Connection) -> None:
    """
    Insert sample patient and provider records for demonstration.
    """
    cur = conn.cursor()

    # Sample patient
    cur.execute(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?)",
        ("Patient_001", 30, "M", "migraine,hypertension", 4.5, 3.0)
    )

    # Sample providers
    providers_sample = [
        ("Doctor_001", "Neurologist", "migraine", 4.8, 0.95, 0.8, 0.2),
        ("Doctor_002", "Cardiologist", "hypertension", 4.5, 0.90, 0.7, 0.4),
        ("Doctor_003", "Pulmonologist", "asthma", 4.2, 0.85, 0.6, 0.5)
    ]
    cur.executemany(
        "INSERT INTO providers VALUES (?,?,?,?,?,?,?)",
        providers_sample
    )

    conn.commit()


def fetch_patient(conn: sqlite3.Connection, patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single patient by ID. Returns None if not found.
    """
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM patients WHERE patient_id=?",
        (patient_id,)
    ).fetchone()
    return dict(row) if row else None


def fetch_all_providers(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Fetch all providers from the database.
    """
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM providers").fetchall()
    return [dict(row) for row in rows]


def seed_database() -> None:
    """
    Full database initialization: create tables and seed sample data.
    """
    conn = get_connection()
    create_tables(conn)
    seed_sample_data(conn)
    conn.close()
    print("✅ Database 'healthcare.db' initialized and seeded with Patient_001!")


if __name__ == "__main__":
    seed_database()