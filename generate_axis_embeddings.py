import os
import pickle
import argparse
import pandas as pd
from pathlib import Path
from llm import request_to_embed, request_to_local_embed
from embed_items import MODELS  # 同じモデル一覧を共有

def main():
    parser = argparse.ArgumentParser(description="キーワードを複数モデルで埋め込み")
    parser.add_argument("folder", help="data 配下のサブフォルダ名 (例: sample, overflow)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "data" / args.folder
    base_dir.mkdir(parents=True, exist_ok=True)
    keyword_path = base_dir / "keyword.csv"
    if not keyword_path.exists():
        raise FileNotFoundError(f"キーワードファイルが見つかりません: {keyword_path}")

    df = pd.read_csv(keyword_path)
    keywords = df["keyword" if "keyword" in df.columns else "キーワード"].dropna().unique().tolist()

    for model_name in MODELS:
        model_key = model_name.replace("/", "_")
        cache_path = base_dir / f"embed_cache_{model_key}.pkl"
        out_path = base_dir / f"keyword_embed_{model_key}.pkl"

        # キャッシュ読み込み
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                embed_cache = pickle.load(f)
        else:
            embed_cache = {}

        # 埋め込み取得
        results = {}
        for kw in keywords:
            if kw in embed_cache:
                vec = embed_cache[kw]
                print(f"✅ キャッシュ使用: {kw}")
            else:
                print(f"🆕 埋め込み取得: {kw}")
                if model_name.startswith("openai/"):
                    vec = request_to_embed([kw], model_name.replace("openai/", ""))[0]
                else:
                    vec = request_to_local_embed([kw], model_name)[0]
                embed_cache[kw] = vec
            results[kw] = vec

        # キャッシュ保存
        with open(cache_path, "wb") as f:
            pickle.dump(embed_cache, f)

        # 結果保存
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

        print(f"✅ 出力完了: {out_path}")

if __name__ == "__main__":
    main()
