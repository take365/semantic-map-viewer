import argparse
import os
import pickle
import pandas as pd
from llm import request_to_local_embed, request_to_embed

# 対応するローカルモデルおよびOpenAIモデルのリスト
MODELS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sbintuitions/sarashina-embedding-v1-1b",
    "cl-nagoya/ruri-v3-310m",
    "openai/text-embedding-3-large",
]


def main():
    parser = argparse.ArgumentParser(description="フォルダ名を指定して埋め込みを実行します")
    parser.add_argument(
        "folder",
        help="data 配下のサブフォルダ名 (例: overflow, sample)",
    )
    args = parser.parse_args()

    # データフォルダパス
    base_dir = os.path.join(os.path.dirname(__file__), "data", args.folder)
    input_csv = os.path.join(base_dir, "args.csv")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"指定された CSV が見つかりません: {input_csv}")

    # CSV 読み込み ("argument" カラムを想定)
    df = pd.read_csv(input_csv)
    texts = df["argument"].astype(str).tolist()

    for model_name in MODELS:
        print(f"📦 モデル {model_name} で埋め込み中...")
        try:
            vectors = []
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    print(f"  🔄 {i}/{len(texts)} 件目を処理中...")
                if model_name.startswith("openai/"):
                    vec = request_to_embed(text, model_name.replace("openai/", ""))[0]
                else:
                    vec = request_to_local_embed(text, model_name)
                vectors.append(vec)

            out_path = os.path.join(base_dir, f"embeddings_{model_name.replace('/', '_')}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(vectors, f)
            print(f"✅ 埋め込み結果を保存: {out_path}")

        except Exception as e:
            print(f"❌ モデル {model_name} でエラーが発生しました: {e}")
            continue  # 次のモデルへ進む

    # テキスト＋全モデルの埋め込みを一つにまとめて保存
    combined = {
        "texts": texts,
        "embeddings": {m.replace('/', '_'): None for m in MODELS},
    }
    # 個別ファイルに保存したベクトルをまとめて読み込む
    for model_name in MODELS:
        key = model_name.replace('/', '_')
        with open(os.path.join(base_dir, f"embeddings_{key}.pkl"), "rb") as f:
            combined["embeddings"][key] = pickle.load(f)

    combined_path = os.path.join(base_dir, f"embedded_items_{args.folder}.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"📦 全モデル結果まとめ保存: {combined_path}")


if __name__ == "__main__":
    main()