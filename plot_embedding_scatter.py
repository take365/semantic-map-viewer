#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from llm import request_to_embed

# ─── 設定 ───────────────────────────────────────────
# 1度だけ定義する軸ワード（左, 右, 上, 下）
WORDS = ["宗教", "行政", "文化", "かわいい"] 
# 表示対象カテゴリ（複数対応）
TOPIC = ["動物・魚", "施設", "職業"] #"素材", "料理", "職業", "動物・魚", "施設"

# ─── ファイルパス定義 ─────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ITEMS_PATH  = os.path.join(BASE_DIR, "data", "embedded_items.pkl")
CACHE_PATH  = os.path.join(BASE_DIR, "data", "embed_cache.pkl")
HTML_PATH   = os.path.join(BASE_DIR, "embedding_scatter.html")

# ─── embed_cache 読み込み ───────────────────────────────
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embed_cache = pickle.load(f)
else:
    embed_cache = {}

# ─── 軸語ベクトル取得（キャッシュ or API呼び出し） ───────────
concept_small = {}
concept_large = {}
for word in WORDS:
    # キャッシュ取得
    entry = embed_cache.get(word)
    vec_small = vec_large = None
    if entry and isinstance(entry, tuple) and len(entry) == 2:
        vec_small, vec_large = entry
        vec_small = np.array(vec_small)
        vec_large = np.array(vec_large)
        print(f"word=[{word}] vec_large shape: {vec_small.shape}, vec_large shape: {vec_large.shape}")
    # キャッシュにないか、次元不一致の場合はAPI呼び出し
    # 合計次元3072の誤保存を分割
    if vec_small is None or vec_large is None or \
       (vec_small.ndim == 1 and vec_small.shape[0] != 1536) or \
       (vec_large.ndim == 1 and vec_large.shape[0] != 1536):
        # 再取得
        vec_small = np.array(request_to_embed([word], model="text-embedding-3-small")[0])
        vec_large = np.array(request_to_embed([word], model="text-embedding-3-large")[0])
        # 万が一大きすぎたら分割
        if vec_small.ndim == 1 and vec_small.shape[0] > 1536:
            tmp = vec_small
            vec_small = tmp[:1536]
            vec_large = tmp[1536:]
        # キャッシュに保存
        embed_cache[word] = (vec_small.tolist(), vec_large.tolist())
    concept_small[word] = vec_small
    concept_large[word] = vec_large
# キャッシュを書き戻し
with open(CACHE_PATH, "wb") as f:
    pickle.dump(embed_cache, f)

# ─── 埋め込みデータ読み込み & カテゴリ絞り込み ────────────
with open(ITEMS_PATH, "rb") as f:
    df = pickle.load(f)

df = df[df["カテゴリ"].isin(TOPIC)].copy()
labels = df["絵文字"].fillna("□").tolist()
hover_texts = df["内容"].tolist()
emb_s = np.vstack(df["small"].values)
emb_l = np.vstack(df["large"].values)

# ─── 射影関数 ───────────────────────────────────────
def project(emb_matrix, concept_vecs):
    # X 軸: WORDS[0] ⇔ WORDS[1]
    axis_x = concept_vecs[WORDS[1]] - concept_vecs[WORDS[0]]
    axis_x /= np.linalg.norm(axis_x)
    # Y 軸: WORDS[2] ⇔ WORDS[3]
    axis_y = concept_vecs[WORDS[2]] - concept_vecs[WORDS[3]]
    axis_y /= np.linalg.norm(axis_y)
    return emb_matrix.dot(axis_x), emb_matrix.dot(axis_y)

x_s, y_s = project(emb_s, concept_small)
x_l, y_l = project(emb_l, concept_large)

# ─── 散布図作成 ─────────────────────────────────────
fig = make_subplots(rows=2, cols=1, subplot_titles=("small embedding", "large embedding"))
# small
fig.add_trace(go.Scatter(
    x=x_s, y=y_s, mode="text", text=labels,
    hovertext=hover_texts, textfont=dict(size=16)
), row=1, col=1)
# large
fig.add_trace(go.Scatter(
    x=x_l, y=y_l, mode="text", text=labels,
    hovertext=hover_texts, textfont=dict(size=16)
), row=2, col=1)

# ─── 軸注釈配置（図枠の外側） ─────────────────────
annotations = []
for i in (1, 2):
    xref, yref = f"x{i}", f"y{i}"
    annotations.extend([
        dict(x=-0.18, y=0, text=WORDS[0], showarrow=False, xref=xref, yref=yref, xanchor="right", yanchor="middle", font=dict(size=12)),
        dict(x= 0.18, y=0, text=WORDS[1], showarrow=False, xref=xref, yref=yref, xanchor="left", yanchor="middle", font=dict(size=12)),
        dict(x=0, y=0.25, text=WORDS[2], showarrow=False, xref=xref, yref=yref, xanchor="center", yanchor="bottom", font=dict(size=12)),
        dict(x=0, y=-0.18, text=WORDS[3], showarrow=False, xref=xref, yref=yref, xanchor="center", yanchor="top", font=dict(size=12)),
    ])
fig.update_layout(
    title=f"{TOPIC}カテゴリにおける意味空間 ({WORDS[0]}⇔{WORDS[1]} × {WORDS[2]}⇔{WORDS[3]})",
    annotations=annotations,
    showlegend=False,
    margin=dict(l=50, r=50, t=100, b=50),
    width=800, height=1000
)
# HTML 保存
fig.write_html(HTML_PATH)
print(f"✅ 散布図を {HTML_PATH} に保存しました。 ")
