import os
import pandas as pd
import pickle
from dotenv import load_dotenv

# 既存のembeddingラッパーを使う
from scripts.llm import request_to_embed
# .env 読み込み（必要に応じて調整）
load_dotenv()

# パスの設定
input_path = "data/items.csv"
output_path = "data/embedded_items.pkl"

# モデル
models = ["text-embedding-3-small", "text-embedding-3-large"]

# CSV読み込み
df = pd.read_csv(input_path)

# 内容だけ取り出してリスト化
texts = df["内容"].tolist()

# 埋め込み取得
print("Generating embeddings (small)...")
df["small"] = request_to_embed(texts, model=models[0])

print("Generating embeddings (large)...")
df["large"] = request_to_embed(texts, model=models[1])

# 保存（pickle）
with open(output_path, "wb") as f:
    pickle.dump(df, f)

print(f"✅ 埋め込み済みデータを保存しました: {output_path}")
