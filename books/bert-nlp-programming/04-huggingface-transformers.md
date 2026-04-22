# 第4章 Hugging Face Transformers

## 1. Core Idea
### [4-1 実装環境]
- Google Colaboratory
- [配布コード](https://colab.research.google.com/github/stockmarkteam/bert-book/blob/master)
- ライブラリのバージョン更新や細かな調整を除き、基本的には配布コードの内容に沿って進める。

### [4-2 Transformers]
- **Transformers**：Hugging Face社が提供しているオープンソースライブラリ。BERTをはじめとする、多様なニューラルネットワークを用いた言語モデルが実装されている。多言語の学習済みモデルが利用可能である点も特徴の一つである。本書の配布コードでは、東北大学の研究チームが作成した日本語のBERT学習済みモデルを使用する。処理の流れとしては、まずトークナイザーを用いて文章をトークン化し、そのデータをBERTに入力して出力を得る。

- **トークン化の手順**
1. MeCabを用いて単語に分割する（辞書には基本的にipadicが用いられる）。
2. WordPieceを用いて単語をさらにトークンへ分割する。

## 2. Result
### [1. コードの目的]
配布コードは、共通のインストール・インポートを行う「セットアップ部分」と、モデルを実際に利用する「実装部分」に分かれている。実装部分では、トークナイザーによるトークン化・エンコーディングを行い、その結果をBERTモデルに適用することを目的とする。

### [2. 重要なコード抜粋]

**# 4-4**　`tokenize()`関数
```python
tokenizer.tokenize('明日は自然言語処理の勉強をしよう。')
# 出力: ['明日', 'は', '自然', '言語', '処理', 'の', '勉強', 'を', 'しよ', '##う', '。']
```

**# 4-7**　`encode()`関数
```python
input_ids = tokenizer.encode('明日は自然言語処理의 공부를 하자.')
print(input_ids)
# 出力: [2, 11475, 9, 1757, 1882, 2762, 5, 8192, 11, 2132, 28489, 8, 3]
```

**# 4-8** `convert_ids_to_tokens()`関数
```python
tokenizer.convert_ids_to_tokens(input_ids)
# 出力: ['[CLS]', '明日', 'は', '自然', '言語', '処理', 'の', '勉強', 'を', 'しよ', '##う', '。', '[SEP]']
```

**# 4-14** BERTをGPUに転送
```python
bert = bert.cuda()
```

**# 4-16** データをGPUに転送
```python
encoding = { k: v.cuda() for k, v in encoding.items() }
```

**# 4-19** 推論の実行
```python
with torch.no_grad():
    output = bert(**encoding)
    last_hidden_state = output.last_hidden_state  # 勾配計算を行わないため、メモリ消費と計算時間を抑えられる。
```

**# 4-20** 後処理
```python
last_hidden_state = last_hidden_state.cpu() # CPUに転送
last_hidden_state = last_hidden_state.numpy() # numpy.ndarrayに変換
last_hidden_state = last_hidden_state.tolist() # リストに変換
```

### [3. Q&A]

**Q1. インストール・インポート部分で、なぜBERTだけでなくPyTorch（torch）もインポートするのか？ PyTorchはこのコードでどのような役割を果たすのか？**

A1. 

**Q2. インポート部分に「from transformers import BertJapaneseTokenizer, BertModel」とあるが、なぜトークナイザーとモデルが分けられているのか？ BERTが多言語に対応するためか？ また、モデルはトークン化とエンコーディング以外のニューラルネットワークの機能をすべて備えているのか？**

A2. 

**Q3. 配布コードには記載のない「unidic-lite」というライブラリがないと正常に動作しなかった。unidic-liteとはどのようなライブラリで、このコードではどのような役割を果たすのか？**

A3. 

**Q4. `tokenizer.tokenize()`の結果は表示されるが、`input_ids = tokenizer.encode(...)`のように変数に代入して`print(input_ids)`を行う場合、なぜトークン化の中間結果（文字列のリスト）は表示されないのか？**

A4. 

**Q5. `convert_ids_to_tokens()`関数を呼び出すと、なぜ自動的に改行された形式で出力されるのか？**

A5. 

**Q6. ID列の長さを揃えるために使われる「padding」と「truncation」は、コード内で具体的にどのような役割を果たすのか？**

A6. 

**Q7. テンソル（Tensor）はこのコードにおいてどのような役割を果たすのか？**

A7. 

**Q8. 4-17で`token_type_ids=encoding['token_type_ids']`を含むコードを実行した際、エラーが発生した。その行を削除してもテンソルサイズなどの後続の処理結果に影響がなかったのはなぜか？**

A8. 

## 3. AI Feedback
