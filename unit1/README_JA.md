# ユニット1：Diffusionモデル入門

Hugging Face Diffusionモデルコースのユニット1へようこそ!このユニットでは、拡散モデルの仕組みの基本を学び、
🤗 Diffusers ライブラリを使用して独自のモデルを作成する方法を学びます。

## このユニットを開始する :rocket:

ユニットの手順は以下の通りです:

- 新しい教材がリリースされたときに通知を受けることができるように、[このコースにサインアップ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)していることを確認してください
- 以下の紹介資料と、興味のありそうな追加資料に目を通してください
- 以下の _**Introduction to Diffusers**_ ノートブックで、🤗Diffusersライブラリを使った理論の実践をチェックしてみてください
- ノートブックまたはリンクされたトレーニングスクリプトを使用して、独自のdiffusionモデルをトレーニングし、共有することができます
- (オプション) 最小限のゼロからの実装に興味があり、様々な設計上の決定を検討したい場合は、 _**Diffusion Models from Scratch**_ ノートブックでより深く掘り下げることができます
- (オプション) [このビデオ](https://www.youtube.com/watch?v=09o5cv6u76c)で、このユニットの教材をざっと見てみてください。


:loudspeaker: [Discord](https://huggingface.co/join/discord)に参加するのを忘れないでください。ここでは、`#diffusion-models-class`チャンネルで教材について議論したり、作ったものを共有したりすることができます。

## Diffusionモデルとは何か？

Diffusionモデルとは、「生成モデル」と呼ばれるアルゴリズム群に比較的最近追加されたものです。生成モデリングの目的は、多くの学習例が与えられたときに、画像や音声などのデータを **生成する** ことを学習することです。優れた生成モデルは、学習データを正確にコピーすることなく、それに類似した **多様な** 出力セットを作成します。diffusionモデルはどのようにしてこれを実現するのでしょうか？ここでは、説明のために画像生成のケースに焦点を当ててみましょう。

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>

diffusionモデルの成功の秘密は、diffusionプロセスの反復性にあります。生成はランダムなノイズから始まりますが、出力画像が現れるまで、何段階にもわたって徐々に洗練されていきます。各ステップにおいて、モデルは現在の入力から完全にノイズ除去されたバージョンまでどのように進むかを推定します。しかし、各ステップで小さな変更を加えるだけなので、初期段階（最終的な出力を予測することが非常に難しい段階）でのこの推定値の誤差は、後の更新で修正することができます。

モデルの学習は、他のタイプの生成モデルに比べて比較的簡単です。以下を繰り返します
1) 学習データから画像をいくつか読み込む
2) 様々な量のノイズを加える。このとき、極端にノイズの多い画像と完璧に近い画像の両方を「修正」（ノイズ除去）する方法をモデルがうまく推定できるようにすることが重要です。
3) ノイズがかかったバージョンの入力をモデルに送り込む
4) これらの入力に対して、モデルがどの程度ノイズ除去を行うかを評価する
5) この情報を使ってモデルの重みを更新する
    
学習されたモデルを使って新しい画像を生成するには、まず完全にランダムな入力から始めて、それを繰り返しモデルに与え、モデルの予測に基づいて毎回少しずつ更新していきます。後述しますが、このプロセスを効率化し、できるだけ少ないステップで良い画像を生成できるようにするためのサンプリングのやり方が数多く存在します。

このユニット1では、それぞれのステップをハンズオンノートで詳しく紹介します。ユニット2では、このプロセスをどのように変更し、追加の条件付け（クラスラベルなど）やガイダンスといった手法によって、モデルの出力にさらなる制御を加えることができるかを見ていきます。そしてユニット3と4では、Stable Diffusionと呼ばれる非常に強力なdiffusionモデルを探求します。このモデルは、テキストの説明が与えられたときに画像を生成することができます

## ハンズオンノートブック

この時点で、付属のノートブックに取り掛かるのに十分な知識があるのです!この2つのノートは、同じアイデアを異なる方法で表現しています。

| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to Diffusers                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              |
| Diffusion Models from Scratch                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              |

In _**Introduction to Diffusers**_, we show the different steps described above using building blocks from the diffusers library. You'll quickly see how to create, train and sample your own diffusion models on whatever data you choose. By the end of the notebook, you'll be able to read and modify the example training script to train diffusion models and share them with the world! This notebook also introduces the main exercise associated with this unit, where we will collectively attempt to figure out good 'training recipes' for diffusion models at different scales - see the next section for more info.

In _**Diffusion Models from Scratch**_, we show those same steps (adding noise to data, creating a model, training and sampling) but implemented from scratch in PyTorch as simply as possible. Then we compare this 'toy example' with the diffusers version, noting how the two differ and where improvements have been made. The goal here is to gain familiarity with the different components and the design decisions that go into them so that when you look at a new implementation you can quickly identify the key ideas.

## プロジェクトタイム

さて、基本を押さえたところで、1つまたは複数の拡散モデルをトレーニングしてみましょう!いくつかの提案は、 _**Introduction to Diffusers**_ のノートブックの最後に記載されています。あなたの結果、トレーニングレシピ、発見をコミュニティと共有し、これらのモデルをトレーニングする最良の方法を共同で見つけ出すことができるようにしてください。

## 追加資料

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) is a very in-depth walk-through of the code and theory behind DDPMs with
 maths and code showing all the different components. It also links to a number of papers for further reading.

Hugging Face documentation on [Unconditional Image-Generation](https://huggingface.co/docs/diffusers/training/unconditional_training) for some examples of how to train diffusion models using the official training example script, including code showing how to create your own dataset.

AI Coffee Break video on Diffusion Models: https://www.youtube.com/watch?v=344w5h24-h8

Yannic Kilcher Video on DDPMs: https://www.youtube.com/watch?v=W-O7AZNzbzQ

Found more great resources? Let us know and we'll add them to this list.
