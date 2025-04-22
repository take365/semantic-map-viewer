import pickle
import json
from pathlib import Path

# 入出力ファイルパス
EMBED_KEYWORD_PATH = "data/embed_keyword.pkl"
EMBED_ITEMS_PATH = "data/embedded_items.pkl"
HTML_OUTPUT_PATH = "embedding_explorer.html"

# 固定カテゴリ
CATEGORIES = ["施設", "職業", "素材", "動物・魚", "料理"]

# Load pickle files
with open(EMBED_KEYWORD_PATH, "rb") as f:
    keyword_data = pickle.load(f)

with open(EMBED_ITEMS_PATH, "rb") as f:
    items_df = pickle.load(f)

# JSON変換（embeddingはそのまま浮動小数）
keywords_json = json.dumps(keyword_data, ensure_ascii=False)
items_json = items_df.to_json(orient="records", force_ascii=False)
categories_json = json.dumps(CATEGORIES, ensure_ascii=False)

# ─── HTML テンプレート全体 ──────────────────────────────
html = f"""
<!DOCTYPE html>
<html lang=\"ja\">
<head>
  <meta charset=\"UTF-8\">
  <title>意味空間 Explorer</title>
  <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
  <style>
    body {{ font-family: sans-serif; }}
    .control-panel {{ margin-bottom: 1em; display: flex; flex-direction: column; align-items: center; }}
    .axis-grid {{ display: grid; grid-template-columns: auto auto auto; grid-template-rows: auto auto auto; gap: 0.5em; margin-bottom: 1em; }}
    .axis-grid > * {{ text-align: center; }}
    .checkboxes {{ display: flex; flex-wrap: wrap; gap: 1em; justify-content: center; margin-bottom: 1em; }}
    select, button, label {{ font-size: 1rem; padding: 0.2em; margin: 0.2em; }}
    .description {{ max-width: 800px; margin: 0 auto 1em auto; font-size: 0.95rem; color: #333; }}
  </style>
</head>
<body>
  <h1>意味空間 Explorer</h1>
  <div class="description">
    このツールは、意味的なキーワードをベクトル空間にエンベディングし、2次元に射影することで視覚化する実験的ツールです。
    OpenAIのtext-embeddingモデル（small/large）で取得したベクトルを、指定した4つの軸語（左右・上下）に基づいて計算し、主観的な意味空間を体験できます。
    射影はベクトル間の差に基づく直交2軸で構成され、各点はアイテムの意味的位置を示します。アイテム、軸のワードはエンベディング済の情報を埋め込んでいるため更新でAPI利用料はかかりません。
  </div>

  <div class="control-panel">
    <div class="axis-grid">
      <div></div>
      <div>
        Y軸上: <select id="y0"></select>
      </div>
      <div></div>
      <div>
        X軸左: <select id="x0"></select>
      </div>
      <div></div>
      <div>
        X軸右: <select id="x1"></select>
      </div>
      <div></div>
      <div>
        Y軸下: <select id="y1"></select>
      </div>
      <div></div>
    </div>

    <div class="checkboxes" id="category-box"></div>

    モデル: <select id="model">
      <option value="small">small</option>
      <option value="large">large</option>
    </select>
    <button onclick="updatePlot()">更新</button>
  </div>

  <div id="plot" style="width:90vw; height:80vh;"></div>

  <script>
    const keywordData = {keywords_json};
    const items = {items_json};
    const categories = {categories_json};

    const keys = Object.keys(keywordData);

    function fillSelect(id, options, defaultValue) {{
      const sel = document.getElementById(id);
      sel.innerHTML = "";
      options.forEach(v => {{
        const opt = document.createElement("option");
        opt.value = v; opt.textContent = v;
        if (v === defaultValue) opt.selected = true;
        sel.appendChild(opt);
      }});
    }}

    fillSelect("x0", keys, "甘い");
    fillSelect("x1", keys, "辛い");
    fillSelect("y0", keys, "冷たい");
    fillSelect("y1", keys, "熱い");

    const categoryBox = document.getElementById("category-box");
    categories.forEach(cat => {{
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = cat;
      checkbox.checked = true;
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(cat));
      categoryBox.appendChild(label);
    }});

    document.getElementById("model").value = "large";

    function normalize(vec) {{
      const norm = Math.sqrt(vec.reduce((a,b) => a + b*b, 0));
      return vec.map(x => x / norm);
    }}

    function dot(vec1, vec2) {{
      return vec1.reduce((a,b,i) => a + b * vec2[i], 0);
    }}

    function updatePlot() {{
      const x0 = document.getElementById("x0").value;
      const x1 = document.getElementById("x1").value;
      const y0 = document.getElementById("y0").value;
      const y1 = document.getElementById("y1").value;
      const model = document.getElementById("model").value;

      const selectedCategories = Array.from(document.querySelectorAll("#category-box input:checked")).map(cb => cb.value);
      const axisX = normalize(keywordData[x1][model].map((v, i) => v - keywordData[x0][model][i]));
      const axisY = normalize(keywordData[y0][model].map((v, i) => v - keywordData[y1][model][i]));

      const filtered = items.filter(row => selectedCategories.includes(row["カテゴリ"]));

      const xs = filtered.map(row => dot(row[model], axisX));
      const ys = filtered.map(row => dot(row[model], axisY));
      const texts = filtered.map(row => row.絵文字 || "□");
      const hovers = filtered.map(row => row.内容);

      const trace = {{
        x: xs, y: ys, text: texts, hovertext: hovers,
        mode: "text", type: "scatter", textfont: {{ size: 16 }}
      }};

      Plotly.newPlot("plot", [trace], {{
        title: `モデル=${{model}}`,
        margin: {{ l: 50, r: 50, t: 100, b: 80 }},
        xaxis: {{}}, yaxis: {{}}
      }});
    }}

    updatePlot();  // 初期表示
  </script>
</body>
</html>
"""

# 保存
Path(HTML_OUTPUT_PATH).write_text(html, encoding="utf-8")
print(f"✅ HTML 出力完了: {HTML_OUTPUT_PATH}")
