# 🧠 意味空間ビジュアライザー & 検索システム

## 概要

このプロジェクトは、テキスト情報（食べ物、施設、職業など）をベクトル化し，
意味的な「軸語（例：甘い⇌辛い、冷たい⇌熱い）」に射射して視覚的に分類・探索するツールです。

OpenAIやローカルモデルを使ったエンベディングと、Plotlyによるインタラクティブな可視化を提供します。

---

## 主な特徴

* OpenAI基盤モデル対応 (`text-embedding-3-small`, `text-embedding-3-large`)
* ローカルモデル（sentence-transformers 等）互換
* 任意の軸語に基づく意味空間プロット
* カテゴリフィルタやモデル切り替えで視点換えが可能
* 類似アイテム検索（cos類似度）にも対応

---

## フォルダ構成（例）

```
project/
├── data/
│   └── sample/
│       ├── args.csv                     # 入力コメント
│       ├── keyword.csv                  # 軸語の定義
│       ├── embedded_items_sample.pkl    # テキスト基づくベクトル
│       ├── keyword_embed_*.pkl          # 軸語の方向ベクトル
│       └── embedding_explorer.html      # 視覚化 UI
├── embed_items.py                       # 基本エンベディング
├── generate_axis_embeddings.py          # 軸語ベクトル生成
├── generate_interactive_html.py         # HTML 出力
├── run_search.py                        # 類似検索スクリプト
```

---

## セットアップ

### 1. 🔑 OpenAI APIキー

`.env` または環境変数で下記を指定

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxx
```

---

### 2. ライブラリインストール

```bash
pip install -r requirements.txt
```

---

## 基本フロー: 基本ベクトル化 → 軸語ベクトル → HTML

### 1. 基本テキストのベクトル化

```bash
python embed_items.py sample
```

* 複数モデルによるベクトル化（OpenAI + ローカル）
* 出力: `embedded_items_sample.pkl`, `embeddings_model名.pkl`

---

### 2. 軸語ベクトルの生成

```bash
python generate_axis_embeddings.py sample
```

* `keyword.csv` に定義された軸語を方向ベクトル化
* 出力: `keyword_embed_model.pkl`

---

### 3. 可視化 HTML (意味空間 Explorer)

```bash
python generate_interactive_html.py sample
```

* 出力: `embedding_explorer.html`
* 任意の4軸、モデル、カテゴリに基づく散布図がブラウザ上で表示されます

---

## 入力CSVの例

### args.csv

| argument    | カテゴリ | 絵文字 |
| ----------- | ---- | --- |
| おにぎりが大好きです。 | 料理   | 🍙  |
| 学校で働いています。  | 職業   | 🏫  |

### keyword.csv

| axis | side  | keyword |
| ---- | ----- | ------- |
| 味    | left  | 甘い      |
| 味    | right | 辛い      |
| 温度感  | left  | 冷たい     |
| 温度感  | right | 熱い      |

---

## 類似アイテム検索（オプション）

```bash
python run_search.py
```

* クエリ語とのcos類似度に基づき、類似アイテムをランキング表示
* 出力ファイル：`search_log_YYYYMMDD_*.txt`

---

## 対応済みモデル

* `openai/text-embedding-3-small`
* `openai/text-embedding-3-large`
* `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
* `sbintuitions/sarashina-embedding-v1-1b`
* `cl-nagoya/ruri-v3-310m`

`embed_items.py` の `MODELS` に追記すれば拡張可能

---

## ライセンス

MIT License

---

## 賛助 & PR 歓迎

* 軸語パターンの強化
* 新モデルの追加
* 評価指標（シルエットスコアなど）の導入

不明点や改善アイデアは歓迎です！
