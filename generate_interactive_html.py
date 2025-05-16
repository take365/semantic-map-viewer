import argparse
import pickle
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="embedding_explorer.html を生成")
    parser.add_argument("folder", help="data 配下のサブフォルダ名 (例: sample, overflow)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "data" / args.folder
    item_path = base_dir / f"embedded_items_{args.folder}.pkl"

    with open(item_path, "rb") as f:
        items_data = pickle.load(f)

    args_path = base_dir / "args.csv"
    if args_path.exists():
        df = pd.read_csv(args_path)
    else:
        df = pd.DataFrame({"argument": items_data["texts"]})
        df["カテゴリ"] = "カテゴリA"
        df["絵文字"] = "□"

    items = [
        {
            "内容": df["argument"][i],
            "絵文字": df["絵文字"][i] if "絵文字" in df else "□",
            "カテゴリ": df["カテゴリ"][i] if "カテゴリ" in df else "カテゴリA",
            **{k: v[i] for k, v in items_data["embeddings"].items()}
        }
        for i in range(len(items_data["texts"]))
    ]

    keyword_data = {}
    for model_key in items_data["embeddings"].keys():
        model_path = base_dir / f"keyword_embed_{model_key}.pkl"
        if not model_path.exists():
            print(f"⚠️ keyword_embed_{model_key}.pkl が見つかりません。スキップ。")
            continue
        with open(model_path, "rb") as f:
            emb = pickle.load(f)
        for kw, vec in emb.items():
            keyword_data.setdefault(kw, {})[model_key] = vec

    models = list(items_data["embeddings"].keys())
    categories = sorted(df["カテゴリ"].unique().tolist())
    items_json = json.dumps(items, ensure_ascii=False)
    keyword_json = json.dumps(keyword_data, ensure_ascii=False)
    categories_json = json.dumps(categories, ensure_ascii=False)

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>意味空間 Explorer</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; }}
    .description {{
      max-width: 1000px;
      margin: 0.3em auto 0.5em auto;
      font-size: 0.95rem;
      color: #333;
      text-align: center;
      white-space: nowrap;
      overflow-x: auto;
    }}
    .control-panel {{
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-bottom: 0.5em;
    }}
    .axis-grid {{
      display: grid;
      grid-template-columns: auto auto auto;
      grid-template-rows: auto auto auto;
      gap: 0.2em;
      margin-bottom: 0.4em;
      font-size: 0.95rem;
    }}
    .axis-grid > * {{ text-align: center; }}
    .checkboxes {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.8em;
      justify-content: center;
      margin-bottom: 0.5em;
    }}
    .control-row {{
      display: flex;
      align-items: center;
      gap: 0.5em;
      margin-bottom: 0.3em;
    }}
    select, button, label {{
      font-size: 0.95rem;
      padding: 0.2em;
      margin: 0.1em;
    }}
  </style>
</head>
<body>
  <div class="description">
    意味的なキーワードで意味空間を探索できる可視化ツール
  </div>

  <div class="control-panel">
    <div class="axis-grid">
      <div></div>
      <div>Y軸上: <select id="y0"></select></div>
      <div></div>
      <div>X軸左: <select id="x0"></select></div>
      <div></div>
      <div>X軸右: <select id="x1"></select></div>
      <div></div>
      <div>Y軸下: <select id="y1"></select></div>
      <div></div>
    </div>

    <div class="checkboxes" id="category-box"></div>

    <div class="control-row">
      モデル: <select id="model"></select>
      <button onclick="updatePlot()">更新</button>
    </div>
  </div>

  <div id="plot" style="width:90vw; height:60vh;"></div>

  <script>
    const keywordData = {keyword_json};
    const items = {items_json};
    const categories = {categories_json};
    const keys = Object.keys(keywordData);
    const models = Object.keys(keywordData[keys[0]]);

    const modelSel = document.getElementById("model");
    models.forEach(m => {{
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSel.appendChild(opt);
    }});
    modelSel.value = models[0];

    function fillSelect(id, options, defaultValue) {{
      const sel = document.getElementById(id);
      sel.innerHTML = "";
      options.forEach(v => {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
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

      const xs = [], ys = [], texts = [], hovers = [];
      for (const row of items) {{
        if (!row[model]) continue;
        if (!selectedCategories.includes(row["カテゴリ"])) continue;
        xs.push(dot(row[model], axisX));
        ys.push(dot(row[model], axisY));
        texts.push(row["絵文字"] || "□");
        hovers.push(row["内容"]);
      }}

      const trace = {{
        x: xs, y: ys, text: texts, hovertext: hovers,
        mode: "text", type: "scatter", textfont: {{ size: 16 }}
      }};

      Plotly.newPlot("plot", [trace], {{
        margin: {{ l: 50, r: 50, t: 20, b: 80 }},
        xaxis: {{}}, yaxis: {{}}
      }});
    }}

    updatePlot();
  </script>
</body>
</html>
"""

    out_path = base_dir / "embedding_explorer.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML 出力完了: {out_path}")

if __name__ == "__main__":
    main()
