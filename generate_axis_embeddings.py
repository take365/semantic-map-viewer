import os
import pickle
import pandas as pd
import numpy as np
from scripts.llm import request_to_embed

# ─── ファイルパス ─────────────────────────────
KEYWORD_PATH = "data/keyword.csv"
CACHE_PATH = "data/embed_cache.pkl"
OUTPUT_PATH = "data/embed_keyword.pkl"

# ─── CSV読み込み ────────────────────────────────
df = pd.read_csv(KEYWORD_PATH)
keywords = df["キーワード"].dropna().unique().tolist()

# ─── キャッシュ読み込み or 初期化 ───────────────
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embed_cache = pickle.load(f)
else:
    embed_cache = {}

# ─── 埋め込み処理 ──────────────────────────────
results = {}
for kw in keywords:
    if kw in embed_cache:
        small, large = embed_cache[kw]
        print(f"✅ キャッシュ使用: {kw}")
    else:
        print(f"🆕 埋め込み取得: {kw}")
        small = request_to_embed([kw], model="text-embedding-3-small")[0]
        large = request_to_embed([kw], model="text-embedding-3-large")[0]
        embed_cache[kw] = (small, large)
    results[kw] = {"small": small, "large": large}

# ─── キャッシュ保存 ─────────────────────────────
with open(CACHE_PATH, "wb") as f:
    pickle.dump(embed_cache, f)

# ─── 結果保存（HTML埋め込み用） ───────────────
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(results, f)

print(f"✅ 出力完了: {OUTPUT_PATH}")
