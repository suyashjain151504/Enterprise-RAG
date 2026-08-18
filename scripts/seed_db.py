import argparse
import os
import random
import time
from pathlib import Path

import psycopg2
from loguru import logger

from app.middleware.auth import hash_password


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adv_rag")
MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "seed", "migrations")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "seed", "docs")

DEMO_USERS = [
    ("agent@demo.local", "agent123", False),
    ("admin@demo.local", "admin123", True),
]


def run_migrations(conn: psycopg2.extensions.connection) -> None:
    cur = conn.cursor()
    files = sorted([f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql")])
    for filename in files:
        path = os.path.join(MIGRATIONS_DIR, filename)
        with open(path) as f:
            sql = f.read()
        logger.info("Running migration: {}", filename)
        cur.execute(sql)
    conn.commit()
    cur.close()



def seed_users(conn: psycopg2.extensions.connection) -> None:
    cur = conn.cursor()
    for username, password, is_admin in DEMO_USERS:
        password_hash = hash_password(password)
        cur.execute(
            """
            INSERT INTO users (username, password_hash, is_admin)
            VALUES (%s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                is_admin = EXCLUDED.is_admin
            """,
            (username, password_hash, is_admin),
        )
        logger.info("Seeded user: {} (admin={})", username, is_admin)
    conn.commit()
    cur.close()
    
    
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Seed DB + ingest documents")
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Run migrations + users only; skip vector-store ingestion",
    )
    parser.add_argument(
        "--noise-sample", default="150",
        help="Number of noisy docs to sample (default 150). Use 0 or 'all'.",
    )
    args = parser.parse_args()

    logger.info("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    logger.info("Running migrations...")
    run_migrations(conn)
    logger.info("Seeding demo users...")
    seed_users(conn)
    conn.close()
    logger.info("DB seeding done.")
    
    
if __name__ == "__main__":
    main()