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

Diffusion models are a relatively recent addition to a group of algorithms known as 'generative models'. The goal of generative modeling is to learn to **generate** data, such as images or audio, given a number of training examples. A good generative model will create a **diverse** set of outputs that resemble the training data without being exact copies. How do diffusion models achieve this? Let's focus on the image generation case for illustrative purposes.

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>

iffusionモデルの成功の秘密は、diffusionプロセスの反復性にある。生成はランダムなノイズから始まりますが、出力画像が現れるまで、何段階にもわたって徐々に洗練されていきます。各段階において、モデルは現在の入力から完全にノイズ除去されたバージョンまでどのように進むかを推定します。しかし、各段階で小さな変更を加えるだけなので、初期段階（最終的な出力を予測することが非常に難しい段階）でのこの推定値の誤差は、後の更新で修正することができます。

Training the model is relatively straightforward compared to some other types of generative model. We repeatedly
1) Load in some images from the training data
2) Add noise, in different amounts. Remember, we want the model to do a good job estimating how to 'fix' (denoise) both extremely noisy images and images that are close to perfect.
3) Feed the noisy versions of the inputs into the model
4) Evaluate how well the model does at denoising these inputs
5) Use this information to update the model weights

To generate new images with a trained model, we begin with a completely random input and repeatedly feed it through the model, updating it each time by a small amount based on the model prediction. As we'll see, there are a number of sampling methods that try to streamline this process so that we can generate good images with as few steps as possible.

We will show each of these steps in detail in the hands-on notebooks here in unit 1. In unit 2, we will look at how this process can be modified to add additional control over the model outputs through extra conditioning (such as a class label) or with techniques such as guidance. And units 3 and 4 will explore an extremely powerful diffusion model called Stable Diffusion, which can generate images given text descriptions.

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
