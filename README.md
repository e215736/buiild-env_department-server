
# 研究のために、学科サーバーで環境構築したメモ
Grounding DINOとSegment Anything Modelの複合したコードを使用すると、ローカルで動かすにはすごく処理が重く、
google colaboratoryでもコードを3回くらい回すだけで、GPU制限を使い果たしちゃう。
そこで、[大学のコースの学科サーバー](https://ie.u-ryukyu.ac.jp/syskan/service/singularity/)を使わざるをえなくなった。

GPU使えるようになるまで3週間の環境構築をしました。

# 前提
琉球大学知能情報コース学科サーバのamaneで作業します。　vscodeで作業できるとすごい楽。下準備としてローカル環境も使う。　　

# 参考資料  
レポジトリ内のg-sam.pyはほとんどこのnotebookをpythonファイルに変換し、実行可能にしたもの。  
+ 実行したいファイルの元：[Combining Grounding DINO with Segment Anything (SAM) for text-based mask generation](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb)

学科サーバー実行手順参考：  
+ [学科サーバで、OpenCALM-3BをLoRAでFine tuningして対話ができるようにしてみる](https://github.com/naltoma/open-calm-finetuning)　　
+ [学科のGPU環境を使う流れ（PyTorch編）](https://scrapbox.io/ie-ryukyu/%E5%AD%A6%E7%A7%91%E3%81%AEGPU%E7%92%B0%E5%A2%83%E3%82%92%E4%BD%BF%E3%81%86%E6%B5%81%E3%82%8C%EF%BC%88PyTorch%E7%B7%A8%EF%BC%89)　　

# 下準備
## 下準備1:必要なversionのライブラリが入ってるdocker image を決める。  
今回決めたimageはこれ＝＞[sorinasmeureanu/cuda11.4-pytorch-py3.8.10-gpu:latest](https://hub.docker.com/layers/sorinasmeureanu/cuda11.4-pytorch-py3.8.10-gpu/latest/images/sha256-8ea9ec2da686bbdebb77fd4dea5d8e2411e499e0ac6d2f73c33b26e12234d765?context=explore)　　
- [docker hub](https://hub.docker.com/)などからイメージを探す。検索欄に必要なライブラリのversionで検索する。(ex.torch 2.1.0,py3.10とか)　　
- ⚠学科サーバーと対応してるtorch,torchaudio,torchvisionのcudaのバージョンがそもそも古いので、結局実行しようとしているファイルが新しめな技術だった場合GPUで実行できない可能性があり、別のversionのものを入れる可能性が高いため、今はなんでもいいはず。　
## 下準備2:ローカル環境で新しい仮想環境を作成し、実行したいファイルが実行できることを確認する。　　
1. 一旦自分のローカルの環境でゼロからpyenv環境(pyenvだとpythonのversionを細かく設定できる)を作成し、実行したいファイルを実行できるように環境を整える。

+ grounding DINOとSAMは新しめな技術なので、必要なライブラリは全て新しめじゃないとエラーで動かなかった。一応、conflictが生じたライブラリがある場合、後にsifファイルで新しく指定したversionを入れ直すこともできるが、ライブラリ同士の要求バージョンを調整するのはかなり骨が折れる作業になった。    

+ ⚠pytorchは学科サーバーのCUDAのバージョンに合わせたpytorchのバージョンをinstallしよう。じゃないと動きません！⚠  
[cuda11.4 pytorch]とかで検索し、いけそうなversionを探そう。     
2. そのあと`pip freeze`コマンドを実行し、その結果をメモしておく。
メモした結果は後でamaneで作業する時に出てくる**calm-ft.def**という定義ファイルに`pip install　libarary_name==version`という形で書き加える。calm-ft.defファイルを見て貰えばわかるはず。　　

# 環境構築手順  
ここから先は全てamaneにログインして、ディレクトリを作ってそこで作業する。　　

## ❶amaneのターミナルで次のコマンドを実行。これでsifファイルを持ってくる。　　
docker://の後に自分が選んだimageのリンクを貼り付ける。　　
今回の場合、cuda11.4-pytorch-py3.8.10-gpu_latest.sifというファイルが生成される。　　  

`singularity pull docker://sorinasmeureanu/cuda11.4-pytorch-py3.8.10-gpu:latest`　


## ❷calm-ft.defファイルを作成し、必要なライブラリに応じて修正・追加する。　
+ [今回自分が作成したdefファイル]{https://github.com/e215736/build-env_department-server/blob/main/calm-ft.def}  
+ ローカルにある、もってきたsifファイルを指定するようにする。
+ 今回自分が作成したdefファイルは、pythonを3.10.13に入れ直し、下準備のpip freezeで出力したライブラリを全てversion指定で記述している。  
+ ❶でもってきたsifファイルのpythonのバージョンは3.8.10だった。ローカルの環境では3.10.13で実行したので、入れ直す作業をしている。3.10.13のimageを探せなかったのでこのような代替案を試した。  
+ またCUDA11.4に対応したpytorchをinstallしないといけないため、以下のような記述をする。
`
    #cuda11.4に対応するtorchをinstallする。

    pip3 uninstall --yes torch torchaudio torchvision

    pip3 install torch==2.0.1 torchaudio torchvision
`   
最新版のtorchではGPUで実行することはできなかったが、
2.0.1だとCUDA11.4に対応し、GPUでファイルを実行することができた。
  
## ❸以下のようにコマンドを実行し、defファイルで記述した追加設定を加えられたsifファイルを生成する。
experiment2.sifはお好きな名前のsifファイルにしてもいい。  

`singularity build --fakeroot experiment2.sif calm-ft.def`　　

## ❹train.sbatchファイルを作成し、適宜パスを合わせる。　
+ 今回作成した[train.sbatch](https://github.com/e215736/build-env_department-server/blob/main/train.sbatch)　
+ `--nv」`はGPU使いますっていう意味。
+ `--cleanenv` オプションは、Singularity コンテナ内の環境変数をクリーンな状態にし、ホスト環境の影響を受けないようにしている。これがなくても実行に影響はなかったけど一応。  

`singularity exec --nv --cleanenv experiment2.sif /usr/bin/env python3.10 /home/student/e21/e215736/experiment2/g-sam.py`　　
　

## ❺logsディレクトリを作成。　　
train.sbatchでファイル実行時に、errorファイルとlogファイルが対で出力される。ちゃんと実行されているかを確認できる。  
+ logファイル：print文やファイルの実行開始や終了が記述される。
+ errorファイル：warning,error,import errorなどが出力される

## ⑥以下のコマンドを実行。　　

`sbatch train.sbatch`　　  

errorファイルにエラーが出てきたらその都度sifファイルの追加設定を修正して正常に動かしたいファイルが実行できるまで繰り返す。  
だめだったら、ベースimageからまた探しに行く。メンタル勝負です。　　

## ⑦GPUで動作しているか確認。これで終わり。
+ [check_GPU.py](https://github.com/e215736/build-env_department-server/blob/main/check_GPU.py)をtrain.sbatchのパスに指定し、GPUがちゃんと起動しているかをチェックする。以下の画像のような結果がlogディレクトリに出力されるのを確認する。

<img width="333" alt="Screenshot 2024-07-08 at 19 48 38" src="https://github.com/e215736/build-env_department-server/assets/93558689/b88d2203-0146-4a42-81b2-d9a0dff46c8f">


+ errorディレクトリには正しくGPUで動かせていた場合何も出力されないが、GPUで認識できていない場合、以下のようなwarningが出る。  

`
/usr/local/lib/python3.10/site-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0
`

