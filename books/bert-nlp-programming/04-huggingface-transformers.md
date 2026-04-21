# Topic
第四章　HuggingFace Transformers

## 1. Core Idea
[4-1 実装環境]
- Google Colaboratory
- 配布コード(https://colab.research.google.com/github/stockmarkteam/bert-book/blob/master)
- ライブラリーのバージョン更新や細かい調整以外は、基本的に配布コード通りに進める。

[4-2 Transformers]
- **Transformers**：Huggingface社が提供しているオープンソースのライブラリー。Bertを含めたさまざまなニューラルネットワークを用いた言語モデルが実装されている。さまざまな言語の事前学習モデルが利用可能であることも特徴の一つである。書籍の配布コードでは東北大学の研究チームによって作成されたBERTの日本語の事前学習モデルを使用する。Transformersはトークナイザーを用いて文章をトークン化し、処理されたデータをBERTに入力して出力を得るという順番で進める。

- **トークン化方法**
1. MeCabを用いて単語に分割する。(MeCabの辞書として基本的にはipadicが用いられる)
2. WordPieceを用いて単語をトークンに分割する。

- Bertでデータを処理する場合、

## 2. Result
[出力の一部]
4-4
tokenizer.tokenize('明日は自然言語処理の勉強をしよう。')
['明日', 'は', '自然', '言語', '処理', 'の', '勉強', 'を', 'しよ', '##う', '。']

