
# 研究のために、学科サーバーで環境構築したメモ
Grounding DINOとSegment Anything Modelの複合したコードを使用すると、ローカルで動かすにはすごく処理が重く、
google colaboratoryでもコードを3回くらい回すだけで、GPU制限を使い果たしちゃう。
そこで、[大学のコースの学科サーバー](https://ie.u-ryukyu.ac.jp/syskan/service/singularity/)を使わざるをえなくなった。

GPU使えるようになるまで3週間の環境構築をしました。

# 前提
琉球大学知能情報コース学科サーバのamaneで作業します。　vscodeで作業できるとすごい楽。下準備としてローカル環境も使う。　　

# 参考資料
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
+ 今回自分が作成したdefファイル  
`
Bootstrap: localimage
From: /home/student/e21/e215736/experiment2/cuda11.4-pytorch-py3.8.10-gpu_latest.sif
%post
    # Python 3.10.13 のインストール
    apt-get update && apt-get install -y wget build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
    libbz2-dev liblzma-dev
    cd /tmp
    wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
    tar xzf Python-3.10.13.tgz
    cd Python-3.10.13
    ./configure --enable-optimizations
    make altinstall

    # pip のアップグレード
    python3.10 -m pip install --upgrade pip

    #cuda11.4に対応するtorchをinstallする。
    pip3 uninstall --yes torch torchaudio torchvision
    pip3 install torch==2.0.1 torchaudio torchvision

    # 必要なパッケージのインストール
    python3.10 -m pip install certifi==2024.6.2 charset-normalizer==3.3.2 contourpy==1.2.1 \
    cycler==0.12.1 filelock==3.15.4 fonttools==4.53.0 fsspec==2024.6.1 huggingface-hub==0.23.4 \
    idna==3.7 Jinja2==3.1.4 kiwisolver==1.4.5 MarkupSafe==2.1.5 matplotlib==3.9.0 mpmath==1.3.0 \
    networkx==3.3 numpy==1.26.4 opencv-python==4.10.0.84 packaging==24.1 pandas==2.2.2 \
    pillow==10.3.0 plotly==5.22.0 pyparsing==3.1.2 python-dateutil==2.9.0.post0 pytz==2024.1 \
    PyYAML==6.0.1 regex==2024.5.15 requests==2.32.3 safetensors==0.4.3 six==1.16.0 sympy==1.12.1 \
    tenacity==8.4.2 tokenizers==0.19.1  tqdm==4.66.4 typing_extensions==4.12.2 \
    tzdata==2024.1 urllib3==2.2.2 transformers@git+https://github.com/huggingface/transformers@1c68f2cafb4ca54562f74b66d1085b68dd6682f5
  
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
2.0.1だとCUDA11.4に対応し、GPUでファイルを実行することができた。GPUで認識できていない場合、以下のようなwarningが出ていた。  

`
/usr/local/lib/python3.10/site-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0
`
  
## ❸以下のようにコマンドを実行し、defファイルで記述した追加設定を加えられたsifファイルを生成する。
experiment2.sifはお好きな名前のsifファイルにしてもいい。  

`singularity build --fakeroot experiment2.sif calm-ft.def`　　

## ❹train.sbatchファイルを作成し、適宜パスを合わせる。　　
+ 「--nv」はGPU使いますっていう意味。　  

`singularity exec --nv --cleanenv experiment2.sif /usr/bin/env python3.10 /home/student/e21/e215736/experiment2/check_GPU.py`　　
　

## ❺logsディレクトリを作成。　　

+ errorファイルとlogファイルが対で出力される。これでちゃんと実行されているかを確認するため。  
+

## ⑥以下のコマンドを実行。これで終わり。　　

`sbatch train.sbatch`　　  

errorファイルにエラーが出てきたらその都度sifファイルの追加設定を修正して正常に動かしたいファイルが実行できるまで繰り返す。  
だめだったら、ベースimageからまた探しに行く。メンタル勝負です。　　

