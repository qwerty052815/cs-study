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
[1. コードの目的]
配布コードはモデルのinstall・import部分とモデルを実際利用する部分に分割できる。


[2. 重要な一部分]
# 4-4　関数tokenize()
tokenizer.tokenize('明日は自然言語処理の勉強をしよう。')
['明日', 'は', '自然', '言語', '処理', 'の', '勉強', 'を', 'しよ', '##う', '。']

# 4-7　関数encode()
input_ids = tokenizer.encode('明日は自然言語処理の勉強をしよう。')
print(input_ids)
[2, 11475, 9, 1757, 1882, 2762, 5, 8192, 11, 2132, 28489, 8, 3]

# 4-8　関数convert_ids_to_tokens()
tokenizer.convert_ids_to_tokens(input_ids)


[3. Q&A]
Q1. install・import部分で何故Bertだけではなく、torchもインポートするのか？torchはこのコードでどの機能を果たすのか？
A1. 

Q2. install・import部分に以下のコードがあるが、何故トークナイザーとモデルが分けられているのか？Bertが多言語対応をするためなのか？逆に、モデルはトークン化とエンコーディング以外のニューラルネットワークの全ての機能を持っているのか？
「from transformers import BertJapaneseTokenizer, BertModel」
A2. 

Q3. 配布コードには存在しない「unidic-lite」というライブラリーがないとコードが正常に作動しなかった。unidic-liteはどのようなライブラリーで、このコードでどういう機能を果たすのか？

Q4. 文章を入力するとトークンのリストを出力する関数tokenizer()はなぜ以下のようにプリンター命令があるコードがある場合は出力されないのか？
tokenizer.tokenize('機械学習を中国語にすると机器学习だ。')　→　出力可能
input_ids = tokenizer.encode('明日は自然言語処理の勉強をしよう。')
print(input_ids)　→　トークナイザーの結果は出力不可能



## 3. AI Feedback
