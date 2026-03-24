# Topic
第四章　HuggingFace Transformers

## 1. Core Idea
[4-1 実装環境]
- Google Colaboratory
- 配布コード(https://colab.research.google.com/github/stockmarkteam/bert-book/blob/master)

[4-2 Transformers]
- **Transformers**：Huggingface社が提供しているオープンソースのライブラリー。Bertを含めたさまざまなニューラルネットワークを用いた言語モデルが実装されている。さまざまな言語の事前学習モデルが利用可能であることも特徴の一つである。書籍の配布コードでは東北大学の研究チームによって作成されたBERTの日本語の事前学習モデルを使用する。Transformersはトークナイザーを用いて文章をトークン化し、処理されたデータをBERTに入力して出力を得るという順番で進める。
