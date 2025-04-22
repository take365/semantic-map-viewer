# 🧠 意味空間ビジュアライザー & 検索システム

## 概要

このプロジェクトは、テキスト情報に基づいてアイテム（食べ物、施設、職業など）をベクトル化し、意味的な「軸語（例：宗教⇔行政、文化⇔かわいい）」に射影して散布図として可視化します。また、任意の検索語に対する意味的類似アイテムの検索も可能です。

## 主な特徴

- OpenAI埋め込みモデル（small/large）を用いたベクトル生成
- 指定した「軸語」に沿った意味空間への射影（2軸プロット）
- カテゴリ別アイテムのフィルタと可視化
- 入力語に対する類似アイテムの検索（cos類似度）
- 結果はHTMLで出力、ブラウザ上で閲覧可能

## 使用技術

- Python 3
- OpenAI Embedding API (`text-embedding-3-small`, `text-embedding-3-large`)
- Plotly（散布図描画）
- dotenv（APIキーなど環境設定）
- pandas / numpy / pickle（データ管理）

## ディレクトリ構成

```
project/
├── data/
│   ├── items.csv               # 入力データ（カテゴリ・内容・絵文字）
│   ├── embedded_items.pkl      # 埋め込み済みデータ（自動生成）
│   └── embed_cache.pkl         # ベクトルキャッシュ（自動生成）
├── scripts/
│   ├── llm.py                  # 埋め込みやLLMへのリクエストラッパー
├── plot_embedding_scatter.py   # 散布図作成スクリプト
├── embed_items.py              # エンベディング生成スクリプト
├── run_search.py               # 意味検索スクリプト
└── .env                        # APIキーなどの環境変数設定
```

## セットアップ手順

1. 🔑 APIキー設定（OpenAI）

OpenAIの埋め込みモデルを利用するために、以下の環境変数を設定してください：

```bash
# システム環境変数として設定（推奨）
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

2. 必要なライブラリをインストール

```bash
pip install -r requirements.txt
```

3. `data/items.csv` に「カテゴリ」「内容」「絵文字」列を記入し準備

4. 埋め込み生成

```bash
python embed_items.py
```

5. 散布図作成

```bash
python plot_embedding_scatter.py
# → embedding_scatter.html が出力され、ブラウザで閲覧可能
```

6. 類似アイテム検索

```bash
python run_search.py
# → 検索結果が search_log_*.txt に出力されます
```

## 例

**軸語設定例：**
- X軸：「宗教」⇔「行政」
- Y軸：「文化」⇔「かわいい」

**カテゴリ例：**
- 動物・魚
- 施設
- 職業
- 素材
- 料理

## 補足

- キャッシュ機能により、同じ語のベクトル取得を省略可能
- OpenAI API or Azure API いずれかに対応（`.env`で切り替え）

---

必要に応じて「デモ画像」や「カテゴリ定義一覧」「ライセンス」「参考リンク」も追記可能です。出力形式など調整したい場合はお知らせください。