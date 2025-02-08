import shutil
import pytest
import subprocess
from pathlib import Path

TUTORIALS_DIR = Path("tutorials")
python_scripts = list(TUTORIALS_DIR.rglob("*.py"))

print(f"[DEBUG] Looking for Python scripts in {TUTORIALS_DIR.resolve()}")
print(f"[DEBUG] Found {len(python_scripts)} Python scripts.")

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        pytest.exit("[SUMO ERROR] SUMO is not installed or not in PATH.")

@pytest.mark.parametrize("script_path", python_scripts)
def test_python_script_execution(script_path):
    try:
        script_filename = script_path.name
        result = subprocess.run(
            ["python", script_filename], capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] Script {script_path} failed to execute: {e.stderr}")