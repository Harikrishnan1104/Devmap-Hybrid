"""
Runs the Colab notebook's cells in a clean temporary folder, exactly in order,
to catch mistakes before it is handed over. The only substitution: the real
pretrained model (not downloadable here) is replaced by a tiny stand-in, and
training is shortened. Everything else is the notebook's own code.
"""
import functools
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "colab", "recreate_base_paper.ipynb")


def main():
    with open(NB) as fh:
        nb = json.load(fh)
    work = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(work)
    sys.path.insert(0, work)
    for name in ["devmap", "train_lib"]:
        sys.modules.pop(name, None)
    ns = {"__name__": "__main__"}
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if src.startswith("%%writefile "):
                first, body = src.split("\n", 1)
                fname = first.split()[1]
                with open(fname, "w") as out:
                    out.write(body)
                with open(os.path.join(HERE, fname)) as ref:
                    assert ref.read().strip() == body.strip(), f"{fname} in notebook differs from tested script"
                print(f"cell {i}: wrote {fname} (identical to the tested script)")
                continue
            if "train_lib.run_experiment(" in src:
                # use the stand-in model, then run the notebook's own loop
                import train_lib
                sys.path.insert(0, HERE)
                from test_train_lib import StubTokenizer, make_encoder
                real = train_lib.run_experiment
                train_lib.run_experiment = functools.partial(
                    real, tokenizer=StubTokenizer(), make_encoder=make_encoder, device="cpu",
                    log=lambda *a, **k: None)
                real_settings_runner = train_lib.run_experiment

                def runner(*a, settings=None, **k):   # keep the notebook's own settings, shorten training
                    return real_settings_runner(*a, settings={**(settings or {}), "max_len": 48,
                                                              "epochs": 1, "train_batch": 16}, **k)
                train_lib.run_experiment = runner
                ns["EXAMS"] = ["old", "unseen_program", "unseen_app", "unseen_suite"]
                ns["SHUFFLE_SEEDS"] = [1, 2]
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            print(f"cell {i}: ran OK")
        files = sorted(os.listdir("results"))
        print("result files:", len(files))
        assert len(files) == 2 * (1 + 2 + 2 + 1), files   # 2 machines x (old + 2 shuffles x 2 exams + suite)
        assert all("+fast" in f for f in files), files     # notebook default is the fast mode
        assert os.path.exists("devmap_results.zip")
        print("NOTEBOOK CHECK PASSED")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
