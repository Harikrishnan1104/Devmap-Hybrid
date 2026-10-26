"""The large per-row prediction files are stored gzipped (results/*.csv.gz) to fit GitHub's
upload limits. Run this once before re-running any analysis script; it writes the .csv files
next to them."""
import glob, gzip, os, shutil
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
for gz in glob.glob(os.path.join(R, "**", "*.csv.gz"), recursive=True):
    out = gz[:-3]
    if not os.path.exists(out):
        with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        print("unpacked", os.path.relpath(out, R))
print("done")
