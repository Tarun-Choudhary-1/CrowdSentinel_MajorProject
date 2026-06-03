
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'users.db')


class User(UserMixin):

    def __init__(self, id_: int, username: str, email: str):
        self.id       = id_
        self.username = username
        self.email    = email

    def get_id(self) -> str:
        return str(self.id)


class UserManager:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    
    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now'))
            )
        """)
        con.commit()
        con.close()

        self._ensure_default_user()

    def _ensure_default_user(self):
    
        con = sqlite3.connect(self.db_path)
        row = con.execute("SELECT COUNT(*) FROM users").fetchone()
        if row[0] == 0:
            pw_hash = generate_password_hash("admin123")
            try:
                con.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
                    ("admin", "admin@crowdsentinel.local", pw_hash)
                )
                con.commit()
                print("[Auth] Default user created: admin / admin123")
            except sqlite3.IntegrityError:
                pass
        con.close()


    def register(self, username: str, email: str,
                 password: str) -> tuple[bool, str]:
    
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        if len(username) < 3:
            return False, "Username must be at least 3 characters."

        pw_hash = generate_password_hash(password)
        try:
            con = sqlite3.connect(self.db_path)
            con.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
                (username.strip(), email.strip().lower(), pw_hash)
            )
            con.commit()
            con.close()
            print(f"[Auth] New user registered: {username}")
            return True, "ok"
        except sqlite3.IntegrityError as e:
            if "username" in str(e).lower():
                return False, "Username already taken."
            if "email" in str(e).lower():
                return False, "Email already registered."
            return False, "Registration failed."

    
    def authenticate(self, username_or_email: str,
                     password: str) -> "User | None":
        
        val = username_or_email.strip()
        val_lower = val.lower()

        con = sqlite3.connect(self.db_path)
        # Try exact username match first, then case-insensitive email match
        row = con.execute(
            "SELECT id, username, email, password_hash FROM users "
            "WHERE username=? OR LOWER(username)=? OR LOWER(email)=?",
            (val, val_lower, val_lower)
        ).fetchone()
        con.close()

        if row is None:
            print(f"[Auth] Login failed - user not found: {val}")
            return None

        if check_password_hash(row[3], password):
            print(f"[Auth] Login success: {row[1]}")
            return User(row[0], row[1], row[2])
        else:
            print(f"[Auth] Login failed - wrong password for: {row[1]}")
            return None

    def get_by_id(self, user_id: int) -> "User | None":
        con = sqlite3.connect(self.db_path)
        row = con.execute(
            "SELECT id, username, email FROM users WHERE id=?",
            (int(user_id),)
        ).fetchone()
        con.close()
        return User(row[0], row[1], row[2]) if row else None
