import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ――― 設定 ―――
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_PATH = os.path.join(BASE_DIR, "data", "embeddings.pkl")
ARGS_PATH = os.path.join(BASE_DIR, "data", "args.csv")
KEYWORD_PATH = os.path.join(BASE_DIR, "data", "embed_keyword.pkl")
HTML_PATH = os.path.join(BASE_DIR, "embedding_scatter.html")

# ――― データ読み込み ―――
df_embed = pd.read_pickle(EMBED_PATH).set_index("arg-id")
df_args = pd.read_csv(ARGS_PATH).set_index("arg-id")
df = pd.concat([df_args, df_embed], axis=1)
df = df[df["embedding"].apply(lambda v: isinstance(v, (list, np.ndarray)) and len(v) == 1536)].copy()

# プロット用データのJSON化
plot_data_json = json.dumps(
    [{"id": idx, "arg": row["argument"], "vec": row["embedding"]}
     for idx, row in df.iterrows()], ensure_ascii=False)

# キーワードベクトル読み込み (smallのみ)
with open(KEYWORD_PATH, "rb") as f:
    keyword_data = pickle.load(f)
keyword_vecs = {k: (v.get("small") if isinstance(v, dict) and "small" in v else v)
                for k, v in keyword_data.items()}
keyword_vecs_json = json.dumps(keyword_vecs, ensure_ascii=False)

# 軸語オプション作成 (入力順を保つ)
def make_options(default):
    opts = []
    for k in keyword_vecs.keys():
        if k == default:
            opts.append("<option value='{0}' selected>{0}</option>".format(k))
        else:
            opts.append("<option value='{0}'>{0}</option>".format(k))
    return "".join(opts)

options_x0 = make_options("現実的")
options_x1 = make_options("理想的")
options_y0 = make_options("革新的")
options_y1 = make_options("危険")

# HTML生成 (.formatを使用しJSの波括弧を二重波括弧でエスケープ)
html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>意味空間コメントプロット</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    .axis-grid {{ display: grid; grid-template-columns: auto auto auto; grid-template-rows: auto auto auto; column-gap: 0.5em; row-gap: 0.5em; width: 800px; margin: 0 auto 1em; text-align: center; }}
  </style>
</head>
<body>
  <h2>意味空間コメントプロット</h2>
  <div style="width:800px; ">
  <div class="axis-grid">
    <div></div>
    <div>Y軸上<br><select id="y0">{options_y0}</select></div>
    <div></div>
    <div>X軸左<br><select id="x0">{options_x0}</select></div>
    <div></div>
    <div>X軸右<br><select id="x1">{options_x1}</select></div>
    <div></div>
    <div>Y軸下<br><select id="y1">{options_y1}</select></div>
    <div></div>
  </div>
  <div style="text-align:center; margin-top:0.5em;"><button onclick="plot()">更新</button></div>
</div>
  <div id="plot" style="width:800px;height:800px;"></div>
  <script>
    const data = {plot_data_json};
    const keywordVecs = {keyword_vecs_json};
    function dot(a, b) {{ return a.reduce((s, ai, i) => s + ai * b[i], 0); }}
    function norm(v) {{ return Math.sqrt(dot(v, v)); }}
    function subtract(a, b) {{ return a.map((ai, i) => ai - b[i]); }}
    function normalize(v) {{ const n = norm(v); return v.map(x => x / n); }}
    function project(vec, ax, ay) {{ return [dot(vec, ax), dot(vec, ay)]; }}
    function plot() {{
      const x0 = keywordVecs[document.getElementById('x0').value];
      const x1 = keywordVecs[document.getElementById('x1').value];
      const y0 = keywordVecs[document.getElementById('y0').value];
      const y1 = keywordVecs[document.getElementById('y1').value];
      const axisX = normalize(subtract(x1, x0));
      const axisY = normalize(subtract(y0, y1));
      const xs = [], ys = [], texts = [];
      for (const d of data) {{
        const [px, py] = project(d.vec, axisX, axisY);
        xs.push(px); ys.push(py);
        texts.push(d.arg.slice(0, 30));
      }}
      Plotly.newPlot('plot', [{{
        x: xs,
        y: ys,
        mode: 'text',
        type: 'scatter',
        text: Array(texts.length).fill('●'),
        hovertext: texts,
        textfont: {{"size": 14}}
      }}], {{
        margin: {{"l": 40, "r": 40, "t": 80, "b": 40}},
        width: 800,
        height: 800
      }});
    }}
    plot();
  </script>
</body>
</html>
""".format(
    options_x0=options_x0,
    options_x1=options_x1,
    options_y0=options_y0,
    options_y1=options_y1,
    plot_data_json=plot_data_json,
    keyword_vecs_json=keyword_vecs_json
)
# 保存
Path(HTML_PATH).write_text(html_template, encoding='utf-8')
print(f"✅ HTML を保存しました: {HTML_PATH}") 
