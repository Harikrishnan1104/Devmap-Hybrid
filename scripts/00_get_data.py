"""Step 1 of the plan: fetch the DevMap dataset (both answer sheets) from the
base paper's public repository into ../data."""
import os
import shutil
import subprocess
import tempfile

REPO = "https://github.com/marco-siino/GM_SOURCE_CODE.git"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FILES = ["AMD_DS/dataset-devmap-amd.csv", "NVIDIA_DS/dataset-devmap-nvidia.csv"]

os.makedirs(DATA, exist_ok=True)
tmp = tempfile.mkdtemp()
try:
    subprocess.run(["git", "clone", "--depth", "1", "-q", REPO, tmp], check=True)
    for f in FILES:
        shutil.copy(os.path.join(tmp, f), DATA)
        print("saved", os.path.basename(f))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
