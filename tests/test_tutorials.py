import os
import shutil
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

TUTORIALS_DIR = Path("../tutorials")

notebooks = list(TUTORIALS_DIR.rglob("*.ipynb"))

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        pytest.exit("SUMO is not installed or not in PATH.")

@pytest.mark.parametrize("notebook_path", notebooks)
def test_notebook_execution(notebook_path): 
    with open(notebook_path, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    executor = ExecutePreprocessor(timeout=1200, kernel_name="python3")

    try:
        executor.preprocess(notebook, {"metadata": {"path": notebook_path.parent}})
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {e}")