import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from llm import request_to_embed
from dotenv import load_dotenv

load_dotenv()

# --- 設定 ---
SEARCH_QUERY = "日本のおいしい食べ物"
TOP_K = 10
FILTER_CATEGORIES = []  # 空リストで「全カテゴリ」
USE_CACHE = True

# --- パス定義 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/embedded_items.pkl")
CACHE_PATH = os.path.join(BASE_DIR, "data/embed_cache.pkl")
LOG_PATH = os.path.join(BASE_DIR, f"search_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# --- キャッシュ読み込み ---
if USE_CACHE and os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embed_cache = pickle.load(f)
else:
    embed_cache = {}

# --- クエリベクトル取得（small + large） ---
if SEARCH_QUERY in embed_cache:
    query_vec_small, query_vec_large = embed_cache[SEARCH_QUERY]
else:
    query_vec_small = request_to_embed([SEARCH_QUERY], model="text-embedding-3-small")[0]
    query_vec_large = request_to_embed([SEARCH_QUERY], model="text-embedding-3-large")[0]
    embed_cache[SEARCH_QUERY] = (query_vec_small, query_vec_large)
    if USE_CACHE:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(embed_cache, f)

# --- 埋め込みデータ読み込み ---
with open(DATA_PATH, "rb") as f:
    df = pickle.load(f)

# --- 類似度計算関数 ---
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- カテゴリでフィルタ（あれば） ---
if FILTER_CATEGORIES:
    df = df[df["カテゴリ"].isin(FILTER_CATEGORIES)]

# --- 類似度計算 ---
df["一致率_small"] = df["small"].apply(lambda vec: cosine_similarity(query_vec_small, vec))
df["一致率_large"] = df["large"].apply(lambda vec: cosine_similarity(query_vec_large, vec))

# --- 出力整形 ---
df["一致率_small"] = (df["一致率_small"] * 100).round(1)
df["一致率_large"] = (df["一致率_large"] * 100).round(1)

# --- smallでソートした上位K件 ---
results_small = df.sort_values("一致率_small", ascending=False).head(TOP_K).copy()

# --- largeでソートした上位K件 ---
results_large = df.sort_values("一致率_large", ascending=False).head(TOP_K).copy()

# --- ログ出力 ---
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"🔍 検索ワード: {SEARCH_QUERY}\n")
    f.write(f"🎯 対象カテゴリ: {'全て' if not FILTER_CATEGORIES else FILTER_CATEGORIES}\n")
    
    f.write(f"\n   small\n\n")
    f.write(f"{'カテゴリ':<8} {'内容':<14} {'small(%)':>10}\n")
    f.write("-" * 42 + "\n")
    for _, row in results_small.iterrows():
        f.write(f"{row['カテゴリ']:<8} {row['内容']:<14} {row['一致率_small']:>10.1f}\n")

    f.write(f"\n   large\n\n")
    f.write(f"{'カテゴリ':<8} {'内容':<14} {'large(%)':>10}\n")
    f.write("-" * 42 + "\n")
    for _, row in results_large.iterrows():
        f.write(f"{row['カテゴリ']:<8} {row['内容']:<14} {row['一致率_large']:>10.1f}\n")

print(f"✅ 結果を {LOG_PATH} に出力しました。")
