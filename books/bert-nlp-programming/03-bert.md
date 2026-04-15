# Topic
第三章　BERT

## 1. Core Idea
[3-1 BERTの構造]
- **Attention(注意機構)**：トークンの情報を処理する際に、他のトークンの情報を直接に参照して処理を行う方法。離れた位置のトークンの情報も適切に取り入れることが可能なため、より深く文脈を考慮した分散表現を得れる。
- Transformer Encoder＝モデルTransformerから提案された、Attentionを用いたニューラルネットワーク。それぞれの層は主にMulti-Head AttenionとFeedforward Networkから構成される。
- Transformer＝トークン化によって得られた文章の構成要素
1. **Scaled Dot-Product Attention**＝Transformerでクエリとキーを評価する方法
qi = xiW^Q, ki = xiW^K, vi = xiW^V ai = Σn j=1 ai,jvj
^ai,j = qi・kj / √d
[ai,1, ai,2,…, ai,n] = Softmax(^ai,1, ^ai,2,…, ^ai,n)
(n=トークン数, i=時刻, xi(i=1,2,…,n)=出力のベクトル, qi=クエリ, ki=キー, vi= バリュー, d=ベクトルの次元, ^ai=スコア, ai,1, ai,2,…, ai,n=重み, vstack=行列を縦に結合する関数)
{効率よく計算する方法}
X = vstack(x1,x2,…,xn), Q = vstack(q1,q2,…,qn), K = vstack(k1,k2,…,kn), V = vstack(v1,v2,…,vn), A = vstack(a1,a2,…,an) = Attention(Q, K, V) = Softmax(QK^T/√d)V Q = XW^Q, K = XW^K, V=XW^V
2. **Multi-Head Attention** = Scaled Dot-Product Attentionの拡張版
ai = hstack(ai^(1), ai^(2),…, ai^(h))W^0 A^(l) = vstack(a1^(l), a2^(l),…,an^(l))
A = vstack(a1,a2,…,an) = hstack(A^(1), A^(2),…, A^(h))W^0 A^(l) = Attention(XW(l)^Q, XW(l)^K, XW(l)^V)
(xi=前の層の出力, i=時刻, qi=クエリ, ki=キー, vi= バリュー, ai=出力, ai^(l)=A^(l)= Scaled Dot-Product Attentionの出力, hstack=行列を横に結合する関数, W^0,W(l)^Q,W(l)^K,W(L)^V=パレメータ, A=Multi-Head Attentionの出力)
3. **Residual Connection** = 深い層を持つモデルに対しても、学習が適切に行える yi = xi+ai
4. **Layer Normalization** = 次の処理への入力を正規化するもの。学習が早く収束することが期待される。 LayerNorm(yi) = γ/σ◉(yi-μi)+β (yi=ベクトル, μi=要素の平均, σi=標準偏差, β,γ=yiと同じ次元のベクトルであるパラメータ)
5. **Feedforward Network** FFN(zi) = GELU(ziWi1+b1)W2+b2(zi=それぞれのトークンに対応するベクトル, GELU関数=ReLU関数を滑らかにしたような関数)

[3-2 入力形式]
1. **トークン化**
文章の前後に特殊トークンを加える。
[SEP]＝文章のペアの境界を示す。入力の終わりを示す
[CLS]＝文章の分散表現として用いる。
Ex) 文章：今日の天気は雪だった。質問：今日の天気は？
入力時→ ’[CLS]’, ‘今日’, ‘の’, ‘天気’, ‘は’, ‘雪’, ‘だっ’, ‘た’, ‘。’, ‘[SEP]’, ‘今日’, ‘の’, ‘天気’, ‘は’, ‘？’, ‘[SEP]’
2. **ベクトル化**
ei = ei^T+ei^S+ei^P
（それぞれのトークンを次元mのベクトルに変換してBERTに入力したもの）

[3-3 学習]
1. **事前学習**
ラベルなしデータを用いて汎用的な言語のパターンを学習させるために行われる。以下の2つの方法の組み合わせ。
- **マスク付き言語モデル**：ランダムに選ばれた15%のトークンを[MASK]という特殊トークンに置き換えた文章をBERTに入力[MASK]の位置に元々あったトークンを予測するモデル
Ex) 今日の天気は雪だった。
入力時→ ’[CLS]’, ‘今日’, ‘の’, ‘天気’, ‘は’, ‘[MASK]’, ‘だっ’, ‘た’, ‘。’, ‘[SEP]’
- **Next Sentence Prediction**：[CLS]に対応するBERTの出力を分類器に入力し、入力された2つの文が連続した文章かそうでないかを判断するタスクを用いて学習するモデル。
Ex) 今日の天気は雪だった。明日も寒い。
入力時→ ’[CLS]’, ‘今日’, ‘の’, ‘天気’, ‘は’, ‘雪’, ‘だっ’, ‘た’, ‘。’, ‘[SEP]’, ‘明日’, ‘も’, ‘寒い’, ‘。’, ‘[SEP]’
2. **ファインチューニング**
個別のタスクのラベル付きデータからBERTがそのタスクに特化するように学習する。 


## 2. Draft Questions
1. Layer Normalizationが”次の処理への入力を正規化するもの”だと説明されているが、ネットに調べたら"ニューラルネットワークの表現力の維持、学習の安定化、汎化性能の向上に貢献するもの"とも説明されているされることもあった。この中ような役割の中で、正規化を通じて学習安定化される理由を知りたいと思った。


## 3. Refined Questions
1. Layer Normalizationについて調べた結果、入力および中間表現を正規化し、各次元のスケールと分布を揃えることで、活性値と勾配の分散を抑え、学習の発散を防ぐことを通じて学習が安定化されることが分かった。このことから、
参考資料: Layer Normalization：学習を安定させる「正規化」の仕組みとは？(https://note.com/koshi_ai/n/n9d31fd428053)
2. 

## 4. AI Feedback
- LLMの性能差を言語そのものの差異に直接帰属させているが、実際には学習データ量・品質・ドメイン偏り・トークン化方式・評価方法など複数の要因が影響するため、原因の分離が必要である。（因果関係の単純化に注意）

## 5. Final Evaluation
第ニ章のノートでは、内容を過度に細分化して整理したため、要点を明確に抽出することができなかったという問題が生じたため、書き方を大幅に変更した。
また、次回からはGoogle Colabでの実装がメインになるため、出力結果を分析することを主に記録する必要性があると考えた。
