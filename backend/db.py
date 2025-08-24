import sqlite3
import json
import os

DB_NAME = "chat_app.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL
    )
    """)

    # Chat sessions (like ChatGPT's sidebar items)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Messages (linked to a session)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        role TEXT NOT NULL,  -- "user" or "ai"
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
    )
    """)

    # Documents (RAG content)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT,
        content TEXT,
        embedding BLOB,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()


def seed_db(seed_file: str = "../data/seed_data.json"):
    if not os.path.exists(seed_file):
        print("Seed file not found.")
        return

    conn = get_connection()
    cursor = conn.cursor()

    with open(seed_file, "r") as f:
        data = json.load(f)

    # Users
    for email in data.get("users", []):
        cursor.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (email,))
    
    # Sessions and messages
    for session in data.get("sessions", []):
        user_id = cursor.execute("SELECT id FROM users WHERE email = ?", (session["email"],)).fetchone()
        if not user_id:
            continue

        cursor.execute(
            "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
            (user_id[0], session["title"])
        )
        session_id = cursor.lastrowid

        for msg in session.get("messages", []):
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, msg["role"], msg["content"])
            )

    # Documents
    for doc in data.get("documents", []):
        user_id = cursor.execute("SELECT id FROM users WHERE email = ?", (doc["email"],)).fetchone()
        if user_id:
            cursor.execute(
                "INSERT INTO documents (user_id, title, content, embedding) VALUES (?, ?, ?, ?)",
                (user_id[0], doc["title"], doc["content"], None)
            )

    conn.commit()
    conn.close()
    print("Database seeded successfully.")


if __name__ == "__main__":
    init_db()
    seed_db()
