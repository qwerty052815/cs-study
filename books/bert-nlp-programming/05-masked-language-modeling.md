# 第5章 文章の穴埋め

## 1. Core Idea
### [4-1 実装環境]
- Google Colaboratory
- [配布コード](https://colab.research.google.com/github/stockmarkteam/bert-book/blob/master)
- ライブラリのバージョン更新や細かな調整を除き、基本的には配布コードの内容に沿って進める。

### [4-2 Bertを用いた文章の穴埋め]
- **BertForMaskedLM**：Transformersで提供されているクラス。特殊トークン[MASK]に入るトークンを語彙から予測できる。
- **貪欲法**：穴埋めにおいて膨大な計算を避けるための近似的方法。最初の[MASK]に一番スコアが高いトークンを穴埋めし、穴埋め後の文章に対して、次の[MASK]を同様に穴埋めする方法を繰り返す。文章の大部分が[MASK]であると、意味がある文章を出力しずらいという課題がある。GPTはこの方法で事前学習を行う。
- **ビームサーチ**：[MASK]を一つ穴埋めするたびに、合計スコアが高い10の文章を候補として残しておき、それをもとに次の[MASK]の穴埋めを行い、また合計スコアの高い10の文章を候補を得るということを繰り返す方法。文章の大部分が[MASK]であると、意味がある文章を出力しずらいという課題がある。

## 2. Result
### [1. コードの目的]
配布コードは、共通のインストール・インポートを行う「セットアップ部分」と、モデルを実際に利用する「実装部分」に分かれている。実装部分では、トークナイザーによるトークン化・エンコーディングを行い、その結果をBERTモデルに適用することを目的とする。

### [2. 重要なコード抜粋]

**# 5-5**　
```python
# BERTに入力し、分類スコアを得る。
# 系列長を揃える必要がないので、単にiput_idsのみを入力します。
with torch.no_grad():
    output = bert_mlm(input_ids=input_ids)
    scores = output.logits
```

**# 5-6**
```python
# ID列で'[MASK]'(IDは4)の位置を調べる
mask_position = input_ids[0].tolist().index(4)

# スコアが最も良いトークンのIDを取り出し、トークンに変換する。
id_best = scores[0, mask_position].argmax(-1).item()
token_best = tokenizer.convert_ids_to_tokens(id_best)
token_best = token_best.replace('##', '')

# [MASK]を上で求めたトークンで置き換える。
text = text.replace('[MASK]',token_best)

print(text)
# 出力: 今日は東京へ行く。
```

**# 5-7**
```python
# スコアが上位のトークンとスコアを求める。
mask_position = input_ids[0].tolist().index(4)
topk = scores[0, mask_position].topk(num_topk)
ids_topk = topk.indices # トークンのID
tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk) # トークン
scores_topk = topk.values.cpu().numpy() # スコア
# 出力: ['[CLS]', '明日', 'は', '自然', '言語', '処理', 'の', '勉強', 'を', 'しよ', '##う', '。', '[SEP]']
```

**# 5-8** 貪欲法
```python
# 5-8
def greedy_prediction(text, tokenizer, bert_mlm):
    """
    [MASK]を含む文章を入力として、貪欲法で穴埋めを行った文章を出力する。
    """
    # 前から順に[MASK]を一つづつ、スコアの最も高いトークンに置き換える。
    for _ in range(text.count('[MASK]')):
        text = predict_mask_topk(text, tokenizer, bert_mlm, 1)[0][0]
    return text

text = '今日は[MASK][MASK]へ行く。'
greedy_prediction(text, tokenizer, bert_mlm)
# 出力: 今日は、東京へ行く。
```

**# 5-10** ビームサーチ
```python
def beam_search(text, tokenizer, bert_mlm, num_topk):
    """
    ビームサーチで文章の穴埋めを行う。
    """
    num_mask = text.count('[MASK]')
    text_topk = [text]
    scores_topk = np.array([0])
    for _ in range(num_mask):
        # 現在得られている、それぞれの文章に対して、
        # 最初の[MASK]をスコアが上位のトークンで穴埋めする。
        text_candidates = [] # それぞれの文章を穴埋めした結果を追加する。
        score_candidates = [] # 穴埋めに使ったトークンのスコアを追加する。
        for text_mask, score in zip(text_topk, scores_topk):
            text_topk_inner, scores_topk_inner = predict_mask_topk(
                text_mask, tokenizer, bert_mlm, num_topk
            )
            text_candidates.extend(text_topk_inner)
            score_candidates.append( score + scores_topk_inner )

        # 穴埋めにより生成された文章の中から合計スコアの高いものを選ぶ。
        score_candidates = np.hstack(score_candidates)
        idx_list = score_candidates.argsort()[::-1][:num_topk]
        text_topk = [ text_candidates[idx] for idx in idx_list ]
        scores_topk = score_candidates[idx_list]

    return text_topk

text = "今日は[MASK][MASK]へ行く。"
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')
# 出力: 今日はお台場へ行く。
今日はお祭りへ行く。
今日はゲームセンターへ行く。
今日はお風呂へ行く。
今日はゲームショップへ行く。
今日は東京ディズニーランドへ行く。
今日はお店へ行く。
今日は同じ場所へ行く。
今日はあの場所へ行く。
今日は同じ学校へ行く。
```

### [3. Q&A]

**Q1. インストール・インポート部分で、なぜBERTだけでなくnumpyもインポートするのか？numpyはこのコードでどのような役割を果たすのか？**

A1. 

**Q2. BertとBertForMaskedLMの関係性は？**

A2. 

**Q3. モデルをGPUに載せることはGPUがディープラーニングに特化した演算装置であるため、必要だと思うが、なぜデータまでGPUに載せる必要があるのか？**

A3. 

**Q4. 推論と学習では、モデルのレイヤー計算（forward処理）は同じなのか？それとも一部だけ計算されるのか？**　あ

A4. 

**Q5. ニューラルネットワークにおける「勾配（gradient）」とは何を意味し、どのようにモデルの更新に関与するのか？**

A5. 

**Q6. `with torch.no_grad():`は単なるコードブロック構文なのか？それとも計算の振る舞いを変更する仕組みなのか？**　あ

A6. 

**Q7. `token_best = token_best.replace('##','')`のこの文章はコードでどういう役割を果たすのか？**

A7. 

**Q8. 貪欲法とビームサーチ以外の方法にはどういうものがあるのか？**

A8. 

**Q9. `text_candidates.extend(text_topk_inner)、score_candidates.append( score + scores_topk_inner)`のこの文章はそれぞれコードでどういう役割を果たすのか？**

A9. 

## 3. AI Feedback
#### 構成と論理性の評価
* **ドキュメント構造**: 「実装環境」「コア概念」「実装結果」「Q&A」という構成は、技術ドキュメントの標準的なフレームワークに沿っており、他者が読んでも再現性が高い構成です。
* **技術的整理**: トークン化の手順（MeCab + WordPiece）や、GPUへのデータ転送（`.cuda()`）など、BERTを扱う上での「必須の注意点」が簡潔にまとめられています。

#### Q&Aおよび実装への考察
* **トラブルシューティングの記録**: `unidic-lite` の不足や `token_type_ids` の挙動など、実際に手を動かさないと気づかない微細なエラーに焦点を当てている点は、備忘録として非常に実用的です。
* **Q&Aの質**: 問いの内容自体はNLP学習における「標準的な疑問」ですが、これらを放置せずに言語化した点は評価できます。ただし、Q8の「エラー行を削除しても動いた理由」については、単に「影響がなかった」で済ませず、モデルの入力仕様（単一文と文ペアの差）までセットで理解しておく必要があります。

#### AI活用の妥当性
* **回答生成への利用**: 現時点でのAIによる回答（Answer）は、専門用語の使い方も正確であり、学習の補助ツールとして正しく機能しています。
* **注意点**: AIの回答はあくまで「一般的な仕様」に基づくものです。ライブラリのバージョン（Transformers 4.x以降など）によっては仕様が変わることもあるため、重要な挙動については必ず[公式ドキュメント](https://huggingface.co/docs/transformers/index)を併記する習慣をつけると、より堅牢なログになります。

#### 今後の改善に向けたアドバイス
* **コードブロックの改善**: 実行結果だけでなく、出力されたテンソルの形状（Shape）などもコメントとして残すと、後段のモデル層との接続を理解する助けになります。
* **「なぜこのモデルか」の視点**: 今回は東北大学のモデルを使用していますが、「なぜ数あるモデルの中でこれがデファクトスタンダードなのか」といった背景を少し調べると、今後の研究選定に役立ちます。

## 4. Final Evaluation
