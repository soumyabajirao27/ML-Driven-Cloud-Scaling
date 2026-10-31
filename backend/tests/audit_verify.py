"""Read-only verification: extract external imports from backend/app,
cross-check against requirements.txt, and confirm importability from venv."""
import ast
import os
import sys

BACKEND_DIR = "D:\\PBL\\backend"
APP_DIR = os.path.join(BACKEND_DIR, "app")

external_imports = set()
local_prefixes = {"app"}

for root, _dirs, files in os.walk(APP_DIR):
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), path)
        except Exception as e:
            print(f"PARSE_ERR: {path}: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    external_imports.add(n.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    external_imports.add(node.module.split(".")[0])

external = sorted(m for m in external_imports if m and not m.startswith("app"))

print("=== EXTERNAL IMPORTS FOUND IN backend/app ===")
for m in external:
    print(" ", m)

print("\n=== IMPORTABILITY CHECK (from venv) ===")
for mod in external:
    try:
        __import__(mod)
        print(f"PASS  import {mod}")
    except Exception as e:
        print(f"ERROR import {mod}: {type(e).__name__}: {e}")

print("\n=== DIRECT RUNTIME CHECKS ===")
check_modules = {
    "FastAPI": "fastapi",
    "SQLAlchemy": "sqlalchemy",
    "PyJWT (jwt)": "jwt",
    "pwdlib": "pwdlib",
    "scikit-learn (sklearn)": "sklearn",
    "joblib": "joblib",
    "numpy": "numpy",
    "psycopg / psycopg.binary": "psycopg",
    "argnon2-cffi": "argon2",
}
for label, mod in check_modules.items():
    try:
        m = __import__(mod)
        print(f"PASS  {label} ({mod}) -> {getattr(m, '__version__', 'OK')}")
    except Exception as e:
        print(f"ERROR {label} ({mod}): {type(e).__name__}: {e}")

print("\n=== requirements.txt cross-check ===")
req_path = os.path.join(BACKEND_DIR, "requirements.txt")
# Map of import-name -> pip-name for the packages we care about.
import_to_pip = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "loguru": "loguru",
    "sqlalchemy": "sqlalchemy",
    "psycopg": "psycopg[binary]",
    "alembic": "alembic",
    "pwdlib": "pwdlib[argon2]",
    "jwt": "PyJWT",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "numpy": "numpy",
}
with open(req_path, encoding="utf-8") as fh:
    req_lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
print("requirements.txt entries:", req_lines)
missing = []
for mod in external:
    pip_name = import_to_pip.get(mod, mod)
    if pip_name.lower() not in (r.lower().split("==")[0].split("[")[0] for r in req_lines):
        missing.append((mod, pip_name))
if missing:
    print("MISSING from requirements.txt:")
    for mod, pip in missing:
        print(f"  import '{mod}' -> pip '{pip}'")
else:
    print("All external imports are covered by requirements.txt.")

# Check for duplicates (same pip package name appearing more than once)
names = [r.lower().split("==")[0].split("[")[0] for r in req_lines]
dups = set(n for n in names if names.count(n) > 1)
print("Duplicate package names:", dups if dups else "none")
