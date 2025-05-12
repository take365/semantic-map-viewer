import argparse
import json
import pickle
from pathlib import Path
from llm import request_to_local_embed, request_to_embed

from embed_items import MODELS
# Plotly ライブラリ
import plotly.graph_objects as go
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="フォルダ単位でインタラクティブHTMLを生成します")
    parser.add_argument(
        "folder",
        help="data 配下のサブフォルダ名 (例: overflow, sample)"
    )
    args = parser.parse_args()

    # フォルダパス設定
    base_dir = Path(__file__).parent / "data" / args.folder
    combined_path = base_dir / f"embedded_items_{args.folder}.pkl"
    if not combined_path.exists():
        raise FileNotFoundError(f"埋め込みまとめファイルが見つかりません: {combined_path}")

    # 埋め込み結果読み込み
    with open(combined_path, "rb") as f:
        data = pickle.load(f)
    texts = data["texts"]            # リスト of str
    embeddings = data["embeddings"]  # dict: {model_key: list[vectors]}

    # キーワードペア読み込み（指定フォルダ内から）
    kw_path = base_dir / "keyword.csv"
    if not kw_path.exists():
        raise FileNotFoundError(f"キーワードファイルが見つかりません: {kw_path}")
    kw_df = pd.read_csv(kw_path)
    axis_names = kw_df["axis"].unique().tolist()

    axis_keywords = {
        axis: {
            "left": kw_df[(kw_df["axis"] == axis) & (kw_df["side"] == "left")]["keyword"].tolist(),
            "right": kw_df[(kw_df["axis"] == axis) & (kw_df["side"] == "right")]["keyword"].tolist(),
        }
        for axis in axis_names
    }

    # キーワード埋め込みの生成（左右の平均ベクトル差分で方向ベクトル）
    keyword_embeddings = {}
    for model_key in embeddings.keys():
        model_name = model_key.replace("_", "/", 1)
        cache_path = base_dir / f"keyword_embed_{model_key}.pkl"
        if cache_path.exists():
            print(f"📦 キャッシュ読み込み: {cache_path.name}")
            with open(cache_path, "rb") as f:
                keyword_embeddings[model_key] = pickle.load(f)
            continue

        print(f"🔤 軸ベクトル作成中: {model_name}")
        model_embeds = {}
        for axis in axis_names:
            left_words = axis_keywords[axis]["left"]
            right_words = axis_keywords[axis]["right"]
            if model_name.startswith("openai/"):
                left_vecs = [request_to_embed(kw, model_name.replace("openai/", ""))[0] for kw in left_words]
                right_vecs = [request_to_embed(kw, model_name.replace("openai/", ""))[0] for kw in right_words]
            else:
                left_vecs = [request_to_local_embed(kw, model_name) for kw in left_words]
                right_vecs = [request_to_local_embed(kw, model_name) for kw in right_words]

            #left_vecs = [request_to_local_embed(kw, model_name) for kw in left_words]
            #right_vecs = [request_to_local_embed(kw, model_name) for kw in right_words]
            def average(vecs):
                return [sum(vals)/len(vals) for vals in zip(*vecs)]
            left_avg = average(left_vecs)
            right_avg = average(right_vecs)
            direction_vec = [r - l for r, l in zip(right_avg, left_avg)]
            model_embeds[axis] = direction_vec
        keyword_embeddings[model_key] = model_embeds

        # キャッシュ保存
        with open(cache_path, "wb") as f:
            pickle.dump(model_embeds, f)

    # HTML 出力先
    out_html = base_dir / f"{args.folder}_interactive.html"

    # JSON ペイロード
    payload = {
        "texts": texts,
        "embeddings": embeddings,
        "models": list(embeddings.keys()),
        "axes": axis_names,
        "keyword_embeddings": keyword_embeddings,
        "axis_keywords": axis_keywords,
    }
    json_path = base_dir / f"interactive_payload_{args.folder}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # HTML テンプレート
    html = [
        "<!DOCTYPE html>",
        "<html lang='ja'><head><meta charset='UTF-8'><title>Interactive Scatter</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "#labels-table {",
        "  display: table;",
        "  width: 100%;",
        "  margin-top: 1em;",
        "  font-family: sans-serif;",
        "  font-size: 0.9em;",
        "  color: #333;",
        "  text-align: center;",
        "}",
        "#labels-table div {",
        "  display: table-cell;",
        "  vertical-align: top;",
        "  padding: 0.3em;",
        "}",
        "</style>",
        "</head><body>",
        "<div style='margin-bottom:1em;'>",
        "モデル: <select id='model-select'></select>&nbsp;",
        "X軸テーマ: <select id='x-axis'></select>&nbsp;",
        "Y軸テーマ: <select id='y-axis'></select>&nbsp;",
        "<button onclick='draw()'>描画</button>",
        "</div>",
        "<div id='plot' style='width:100%;height:80vh;'></div>",
        "<div id='labels-table'>",
        "  <div id='y-top'></div>",
        "  <div></div>",
        "</div>",
        "<div id='labels-table'>",
        "  <div id='x-left'></div>",
        "  <div id='x-right' style='text-align: right;'></div>",
        "</div>",
        "<div id='labels-table'>",
        "  <div id='y-bottom'></div>",
        "  <div></div>",
        "</div>",
        "<script>",
        f"const payload = {json.dumps(payload)};",
        "payload.models.forEach(m => document.getElementById('model-select').innerHTML += `<option value='${m}'>${m}</option>`);",
        "payload.axes.forEach(a => {",
        "  document.getElementById('x-axis').innerHTML += `<option value='${a}'>${a}</option>`;",
        "  document.getElementById('y-axis').innerHTML += `<option value='${a}'>${a}</option>`;",
        "});",
        "function draw(){",
        "  const model = document.getElementById('model-select').value;",
        "  const axisX = document.getElementById('x-axis').value;",
        "  const axisY = document.getElementById('y-axis').value;",
        "  const xVec = payload.keyword_embeddings[model][axisX];",
        "  const yVec = payload.keyword_embeddings[model][axisY];",
        "  const dots = payload.embeddings[model];",
        "  const dotX = dots.map(v => v.reduce((acc, val, i) => acc + val * xVec[i], 0));",
        "  const dotY = dots.map(v => v.reduce((acc, val, i) => acc + val * yVec[i], 0));",
        "  const trace = { x: dotX, y: dotY, mode: 'markers', text: payload.texts, type: 'scatter' };",
        "  Plotly.newPlot('plot', [trace], { margin: { t: 30 } });",
        "  const leftX = payload.axis_keywords[axisX].left.join(', ');",
        "  const rightX = payload.axis_keywords[axisX].right.join(', ');",
        "  const topY = payload.axis_keywords[axisY].right.join(', ');",
        "  const bottomY = payload.axis_keywords[axisY].left.join(', ');",
        "  document.getElementById('x-left').innerText = '← ' + leftX;",
        "  document.getElementById('x-right').innerText = rightX + ' →';",
        "  document.getElementById('y-top').innerText = '↑ ' + topY;",
        "  document.getElementById('y-bottom').innerText = bottomY + ' ↓';",
        "}",
        "</script>",
        "</body></html>"
    ]


    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"✅ インタラクティブ HTML を生成しました: {out_html}")


if __name__ == '__main__':
    main()
