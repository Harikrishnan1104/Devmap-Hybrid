"""
Turns each distinct piece of DevMap code into a list of numbers (an
"embedding") using a pretrained model, WITHOUT any training. These numbers
are the model's own summary of what the code looks like and does.

Only the code text goes in - never the job numbers - so the embedding cannot
act as a name tag for the job size.

  embed(...)            runs a model over the 249 distinct kernels, saves an .npz
  load_embeddings(path) reads it back as {code id: vector}
  code_ids(df)          the code id of every dataset row (to look vectors up)

A code id is the MD5 of the whitespace-normalised code (devmap.code_key), so
the same code always gets the same id on any computer.

Two ways of reading a kernel:
  cut      - the first max_len tokens only (what the fine-tuned models do;
             87 of the 249 kernels are longer than 512 tokens and get cut)
  pieces   - the whole kernel, split into pieces of at most max_len tokens;
             the kernel's vector is the average of the pieces' vectors,
             weighted by how many tokens each piece has
"""
import hashlib
import os

import numpy as np
import torch

import devmap

# Each model: where to download it, how many tokens one piece may have, how
# to read long kernels, and model-specific loading details from its model card.
SPECS = {
    # the two models fine-tuned in the main study (first 512 tokens only)
    "codeberta": {"hub": "huggingface/CodeBERTa-language-id", "max_len": 512, "pieces": False},
    "graphcodebert": {"hub": "microsoft/graphcodebert-base", "max_len": 512, "pieces": False},
    # same model, but reading the whole kernel in pieces
    "graphcodebert-pieces": {"hub": "microsoft/graphcodebert-base", "max_len": 512, "pieces": True},
    # newer models (Phase 1)
    "unixcoder": {"hub": "microsoft/unixcoder-base", "max_len": 1024, "pieces": True},
    "codet5p": {"hub": "Salesforce/codet5p-110m-embedding", "max_len": 512, "pieces": True,
                "remote": True},
    "codesage": {"hub": "codesage/codesage-base-v2", "max_len": 1024, "pieces": True,
                 "remote": True, "tokenizer_args": {"add_eos_token": True}},
    "jina-code": {"hub": "jinaai/jina-embeddings-v2-base-code", "max_len": 4096, "pieces": True,
                  "remote": True},
    "codebert": {"hub": "microsoft/codebert-base", "max_len": 512, "pieces": False},
}
MODELS = {k: v["hub"] for k, v in SPECS.items()}       # kept for older scripts


def code_id(src):
    return hashlib.md5(devmap.code_key(src).encode("utf-8")).hexdigest()


def code_ids(df):
    return df["src"].map(code_id).to_numpy()


def unique_codes(data_dir=devmap.DATA_DIR):
    """(ids, texts) for every distinct kernel. The code is the same on both
    machines' answer sheets, so the AMD file is enough."""
    df = devmap.load("amd", data_dir)
    seen, ids, texts = set(), [], []
    for src in df["src"]:
        i = code_id(src)
        if i not in seen:
            seen.add(i)
            ids.append(i)
            texts.append(devmap.code_key(src))
    return ids, texts


def compat_shims():
    """Some models ship their own code written for older versions of the
    transformers library, which import helpers that later versions moved or
    removed. Put those helpers back where the old code looks for them. This only
    makes the old code importable; it does not change any computation."""
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    if not hasattr(mu, "Conv1D") and hasattr(pu, "Conv1D"):          # CodeSage
        mu.Conv1D = pu.Conv1D
    if not hasattr(pu, "find_pruneable_heads_and_indices"):          # jina-code on transformers 5
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def n_tokens(tokenizer, text):
    return len(tokenizer([text], truncation=False)["input_ids"][0])


def split_into_pieces(text, tokenizer, max_len, step=24):
    """Split code into consecutive pieces of at most max_len tokens (special
    tokens included), cutting only between words. Every word lands in exactly
    one piece, in order."""
    words = text.split(" ")
    pieces, current = [], []
    for i in range(0, len(words), step):
        block = words[i:i + step]
        trial = current + block
        if current and n_tokens(tokenizer, " ".join(trial)) > max_len:
            pieces.append(" ".join(current))
            current = list(block)
        else:
            current = trial
    if current:
        pieces.append(" ".join(current))
    return pieces


def _vectors(model, tokenizer, text, max_len, device):
    """(mean vector, first-token vector, tokens used) for one piece of text."""
    enc = tokenizer([text], truncation=True, max_length=max_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device) if "attention_mask" in enc else torch.ones_like(ids)
    try:
        out = model(input_ids=ids, attention_mask=mask)
    except TypeError:                                   # some remote models take ids only
        out = model(ids)
    first = out[0] if not torch.is_tensor(out) else out
    first = first.float()
    if first.dim() == 3:                                # token vectors -> average over real tokens
        m = mask.unsqueeze(-1).float()
        mean = (first * m).sum(1) / m.sum(1).clamp(min=1)
        cls = first[:, 0]
    else:                                               # the model already returns one vector
        mean = cls = first.reshape(1, -1)
    return mean[0].cpu().numpy(), cls[0].cpu().numpy(), int(mask.sum().item())


def embed(model_key, data_dir, out_dir, tokenizer=None, make_encoder=None, device=None,
          max_len=None, pieces=None, log=print):
    """Saves <out_dir>/embeddings__<model_key>.npz with, per kernel:
       mean   - average of the model's token vectors (main choice)
       cls    - the first token's vector
       tokens - tokens actually read;  n_pieces - pieces read
    Skips the work if the file already exists."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"embeddings__{model_key}.npz")
    if os.path.exists(path):
        log(f"{path} already exists, skipping")
        return path
    spec = SPECS.get(model_key, {})
    max_len = max_len or spec.get("max_len", 512)
    pieces = spec.get("pieces", False) if pieces is None else pieces
    if tokenizer is None or make_encoder is None:
        from transformers import AutoModel, AutoTokenizer
        compat_shims()
        hub, remote = spec["hub"], spec.get("remote", False)
        tokenizer = AutoTokenizer.from_pretrained(hub, trust_remote_code=remote, **spec.get("tokenizer_args", {}))
        make_encoder = lambda: AutoModel.from_pretrained(hub, trust_remote_code=remote)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_encoder().to(device).eval()

    ids, texts = unique_codes(data_dir)
    means, clss, tokens, n_pieces = [], [], [], []
    with torch.no_grad():
        for k, text in enumerate(texts):
            # On a small GPU a very long piece may not fit in memory. Then read that
            # kernel again in smaller pieces (halving until it fits), rather than crash.
            limit = max_len
            while True:
                try:
                    parts = split_into_pieces(text, tokenizer, limit) if (pieces or limit < max_len) else [text]
                    vecs = [_vectors(model, tokenizer, p, limit, device) for p in parts]
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if limit <= 128:
                        raise
                    limit //= 2
                    log(f"   kernel {k + 1}: GPU memory full, reading it in pieces of {limit} tokens instead")
            w = np.array([v[2] for v in vecs], dtype=float)
            means.append(np.average(np.stack([v[0] for v in vecs]), axis=0, weights=w))
            clss.append(vecs[0][1])
            tokens.append(int(w.sum()))
            n_pieces.append(len(parts))
            if (k + 1) % 25 == 0 or k + 1 == len(texts):
                log(f"   {model_key}: {k + 1}/{len(texts)} kernels")
    tokens, n_pieces = np.array(tokens), np.array(n_pieces)
    np.savez_compressed(path, ids=np.array(ids), mean=np.stack(means), cls=np.stack(clss),
                        tokens=tokens, n_pieces=n_pieces, model=spec.get("hub", model_key))
    log(f"saved {path}: {len(ids)} kernels, vectors of {len(means[0])} numbers, "
        f"{int((n_pieces > 1).sum())} kernels read in more than one piece")
    return path


def load_embeddings(path, pooling="mean"):
    z = np.load(path, allow_pickle=False)
    return dict(zip(z["ids"].tolist(), z[pooling]))
