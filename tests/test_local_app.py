import os
import py_compile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_project_layout():
    assert os.path.isfile(os.path.join(BASE_DIR, "main.py"))
    assert os.path.isfile(os.path.join(BASE_DIR, "requirements.txt"))
    assert os.path.isdir(os.path.join(BASE_DIR, "src"))


def test_all_python_modules_compile():
    errors = []
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "tests"}]
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                try:
                    py_compile.compile(fp, doraise=True)
                except Exception as exc:
                    errors.append(f"{fp}: {exc}")
    assert not errors, f"Syntax errors: {errors}"
