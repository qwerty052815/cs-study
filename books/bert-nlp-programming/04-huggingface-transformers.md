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

配布コードは共通的にinstall・importをするセットアップ部分とモデルを実際に利用する実装部分に分割できる。実装部分では、トークナイザーを利用した簡単なトークン化・エンコーディングし、その結果をBertモデルに適用することを目的とすると思われる。

[2. 重要な一部分]

**# 4-4**　関数tokenize()
tokenizer.tokenize('明日は自然言語処理の勉強をしよう。')

['明日', 'は', '自然', '言語', '処理', 'の', '勉強', 'を', 'しよ', '##う', '。']

#4-7　関数encode()
input_ids = tokenizer.encode('明日は自然言語処理の勉強をしよう。')
print(input_ids)

[2, 11475, 9, 1757, 1882, 2762, 5, 8192, 11, 2132, 28489, 8, 3]

#4-8 関数convert_ids_to_tokens()
tokenizer.convert_ids_to_tokens(input_ids)

['[CLS]',
 '明日',
 'は',
 '自然',
 '言語',
 '処理',
 'の',
 '勉強',
 'を',
 'しよ',
 '##う',
 '。',
 '[SEP]']

#4-14
bert = bert.cuda()　# BERTをGPUに載せる

#4-16
encoding = { k: v.cuda() for k, v in encoding.items() }　# データをGPUに載せる

#4-19 
with torch.no_grad():
    output = bert(**encoding)
    last_hidden_state = output.last_hidden_state　# 計算の途中過程が保存されず、メモリーや計算時間を減らせる。

#4-20
last_hidden_state = last_hidden_state.cpu() # CPUにうつす。
last_hidden_state = last_hidden_state.numpy() # numpy.ndarrayに変換
last_hidden_state = last_hidden_state.tolist() # リストに変換


[3. Q&A]

Q1. install・import部分で何故Bertだけではなく、torchもインポートするのか？torchはこのコードでどの役割を果たすのか？

A1. 

Q2. install・import部分に以下のコードがあるが、何故トークナイザーとモデルが分けられているのか？Bertが多言語対応をするためなのか？逆に、モデルはトークン化とエンコーディング以外のニューラルネットワークの全ての機能を持っているのか？
「from transformers import BertJapaneseTokenizer, BertModel」

A2. 

Q3. 配布コードには存在しない「unidic-lite」というライブラリーがないとコードが正常に作動しなかった。unidic-liteはどのようなライブラリーで、このコードでどういう役割を果たすのか？

A3. 

Q4. 関数tokenizer()はなぜ以下のようにプリンター命令があるコードがある場合は出力されないのか？
tokenizer.tokenize('機械学習を中国語にすると机器学习だ。')　→　出力可能
input_ids = tokenizer.encode('明日は自然言語処理の勉強をしよう。')
print(input_ids)　→　トークナイザーの結果は出力不可能

A4. 

Q5. 関数convert_ids_to_tokens()はなぜ自動的に改行した結果を出力するのか？

A5. 

Q6. ID列の長さをするために使われたpaddingとtruncationはコードで具体的にどういう役割を果たすのか？

A6. 

Q7. テンソルはこのコードでどういう役割を果たすのか？

A7. 

Q8. 4-17でtoken_type_ids=encoding['token_type_ids']を含んだコードはエラーが出た。その行だけを削除した場合、テンソルのサイズなど、あとの処理結果には変更がなかったのはなぜなのか？

A8. 


## 3. AI Feedback

