"""Offline check of 15_jev_alternatives.py. Real Laya/Kev weights cannot be downloaded here, so this builds tiny
random checkpoints in the exact layouts the two packages read, runs each package's own fine-tuning code on a few
DevMap rows, and checks the predictions come back in the right shape. Plumbing only; says nothing about accuracy."""
import importlib, json, os, shutil, sys, tempfile
import numpy as np, pandas as pd, torch
import devmap

J = importlib.import_module("15_jev_alternatives")
KEV_DIR = os.environ.get("KEV_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kev"))   # git clone https://github.com/jaredpalmer/kev.git


def corpus():
    df = devmap.load("amd")
    return [J.make_state(r, "amd") for _, r in df.head(120).iterrows()]


def tiny_laya(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
    from transformers import PreTrainedTokenizerFast, ModernBertConfig, AutoModel
    from laya.common import DecisionModel
    from laya.train import save_checkpoint
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.train_from_iterator(corpus() + ["Which device runs this OpenCL kernel faster CPU GPU choice question"],
                          trainers.WordPieceTrainer(vocab_size=600, special_tokens=specials))
    t.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                                     special_tokens=[("[CLS]", t.token_to_id("[CLS]")), ("[SEP]", t.token_to_id("[SEP]"))])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    ecfg = ModernBertConfig(vocab_size=t.get_vocab_size(), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=128, max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
                            cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id, global_attn_every_n_layers=1)
    torch.manual_seed(0)
    model = DecisionModel(AutoModel.from_config(ecfg), head_layers=1, n_act=1)
    cfg = {"encoder": "tiny-modernbert", "max_len": 512, "head_max_len": 96, "head_layers": 1, "act_costs": {}}
    path = os.path.join(root, "laya_tiny")
    save_checkpoint(model, tok, cfg, path)
    return path


def tiny_qwen35(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    sys.path.insert(0, KEV_DIR)
    from kev.model import SPECIAL
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(corpus() + ["Which device runs this OpenCL kernel faster CPU GPU"],
                          trainers.BpeTrainer(vocab_size=800, special_tokens=["<|endoftext|>"] + SPECIAL,
                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<|endoftext|>", eos_token="<|endoftext|>",
                                  additional_special_tokens=SPECIAL)
    cfg = Qwen3_5TextConfig(vocab_size=t.get_vocab_size(), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                            max_position_embeddings=2048, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    path = os.path.join(root, "qwen35_tiny")
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def sample(machine, n_study=48, n_exam=12):
    df = devmap.load(machine).set_index("row_id", drop=False)
    r = devmap.build_exam(df, J.EXAM, seed=1)[0]
    rng = np.random.RandomState(0)
    st = df.loc[r["study"]].iloc[rng.permutation(len(r["study"]))[:n_study]]
    ex = df.loc[r["exam"]].iloc[:n_exam]
    return st, ex


def check_laya(root):
    base = tiny_laya(root)
    st, ex = sample("nvidia")
    runner = J.LayaRunner(base=base, epochs=1, micro_batch=8, device="cpu")
    p = runner.fit_predict(st, ex, "nvidia", 1, os.path.join(root, "w_laya"))
    assert p.shape == (len(ex),) and np.all((p >= 0) & (p <= 1)), p
    print(f"laya: package's own finetune + predict_batch on a tiny checkpoint -> {len(p)} GPU probabilities: ok")


def check_kev(root):
    base = tiny_qwen35(root)
    st, ex = sample("amd")
    runner = J.KevRunner(kev_dir=KEV_DIR, base=base, init_from="", epochs=1, accum=4, max_state=384, device="cpu")
    p = runner.fit_predict(st, ex, "amd", 1, os.path.join(root, "w_kev"))
    assert p.shape == (len(ex),) and np.all((p >= 0) & (p <= 1)), p
    print(f"kev: repo's own kev.train + checkpoint load + scoring on a tiny Qwen3.5 -> {len(p)} GPU probabilities: ok")


class Fake:
    label = "fake"
    def __init__(self): self.calls = 0
    def fit_predict(self, study, exam, machine, seed, workdir):
        os.makedirs(workdir, exist_ok=True); self.calls += 1
        assert not set(study.app) & set(exam.app)
        return np.full(len(exam), 0.7)


def check_run(root):
    out = os.path.join(root, "out")
    f = Fake(); J.run([f], machines=("amd",), out_dir=out, work_root=os.path.join(root, "work"), log=lambda *a: None)
    p = J.result_path(out, "fake", "amd", 1); d = pd.read_csv(p)
    assert len(d) == 680 and f.calls == 5 and not os.listdir(os.path.join(root, "work"))
    d[d["round"] != "pile4"].to_csv(p, index=False)
    f2 = Fake(); J.run([f2], machines=("amd",), out_dir=out, work_root=os.path.join(root, "work"), log=lambda *a: None)
    assert f2.calls == 1 and len(pd.read_csv(p)) == 680
    ref = pd.read_csv("../results/colab_all/preds__graphcodebert+fast__amd__unseen_app__split1__train1.csv")
    assert pd.read_csv(p).groupby("round").row_id.apply(frozenset).equals(ref.groupby("round").row_id.apply(frozenset))
    print("run: same piles as graphcodebert+fast, resume redoes only the missing pile, work folders deleted: ok")


if __name__ == "__main__":
    root = tempfile.mkdtemp()
    try:
        check_run(root)
        check_laya(root)
        check_kev(root)
        print("ALL JEV-ALTERNATIVE CHECKS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)
