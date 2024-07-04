# 研究のために、学科サーバーで環境構築したメモ
Grounding DINOとSegment Anything Modelの複合したコードを使用すると、ローカルで動かすにはすごく処理が重く、
google colaboratoryでもコードを3回くらい回すだけで、GPU制限を使い果たしちゃう。
そこで、[大学のコースの学科サーバー](https://ie.u-ryukyu.ac.jp/syskan/service/singularity/)を使わざるをえなくなった。
##　前提
-琉球大学知能情報コース学科サーバのamaneで作業します。
-下準備としてローカル環境も使う。

#　参考資料
実行したいファイルの元：[Combining Grounding DINO with Segment Anything (SAM) for text-based mask generation](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb)
学科サーバー実行手順参考：
[学科サーバで、OpenCALM-3BをLoRAでFine tuningして対話ができるようにしてみる](https://github.com/naltoma/open-calm-finetuning)
[学科のGPU環境を使う流れ（PyTorch編）](https://scrapbox.io/ie-ryukyu/%E5%AD%A6%E7%A7%91%E3%81%AEGPU%E7%92%B0%E5%A2%83%E3%82%92%E4%BD%BF%E3%81%86%E6%B5%81%E3%82%8C%EF%BC%88PyTorch%E7%B7%A8%EF%BC%89)

#　下準備
## 必要なversionのライブラリが入ってるdocker image を決める。
-今回決めたimageはこれ＝＞sorinasmeureanu/cuda11.4-pytorch-py3.8.10-gpu:latest
-[docker hub](https://hub.docker.com/)などからイメージを探す。検索欄に必要なライブラリのversionで検索する。(ex.torch 2.3.0,py3.10とか)
-できるだけ必要なライブラリは全て新しめがいい。ライブラリ同士の要求バージョンを調整するのはかなり骨が折れる。一応、古いライブラリがある場合、後にsifファイルで新しく指定したpythonのversion入れ直すこともできる。
-学科サーバーと対応してるtorch,torchaudio,torchvisionのcudaのバージョン（ex. 11.3.0+cuなど「＋cu」ついてるやつ）がそもそも古いので、結局実行しようとしている内容が新しめの技術だった場合実行できない。torchもとにかく新しめにした方がいいはず。
## ローカル環境で新しい仮想環境を作成し、実行したいファイルが実行できることを確認。
一旦自分のローカルの環境でゼロから仮想環境(venv)を作成し、実行したいファイルを実行できるようにする。そのあと`pip freeze`コマンドを実行し、その結果をcalm-ft.defファイルにpip install　libarary_name==versionという形で書き加える。calm-ft.defファイルを見て貰えばわかる。

#　手順
ここから下は全部amaneにログインして、ディレクトリを作ってそこで作業する。

## 学科のターミナルで以下のコマンドを実行。これでsifファイルを持ってくる。
docker://の後に自分が選んだimageのリンクを貼り付ける。
`singularity pull docker://sorinasmeureanu/cuda11.4-pytorch-py3.8.10-gpu:latest`
今回の場合、cuda11.4-pytorch-py3.8.10-gpu_latest.sifというファイルが生成される。

##　calm-ft.defファイルを作成し、必要なライブラリに応じて修正・追加する。
-ローカルにあるもってきたsifファイルを指定する。
-今回自分が作成したdefファイルは、pythonを3.10.13に入れ直し、下準備のpip freezeで出力したライブラリを全てversion指定で記述している。もってきたsifファイルのpythonのバージョンは3.8.10だった。ローカルの環境では3.10.13で実行したので、入れ直す作業をしている。3.10.13のimageを探せなかったのでこのように代替案を試した。

# 以下のようにコマンドを実行し、sifファイルにdefファイルに記述した追加設定を加える。
`singularity build --fakeroot experiment2.sif calm-ft.def`
-experiment2.sifはお好きな名前のsifファイルにしてもいい。

##　train.sbatchファイルを作成し、適宜パスを合わせる。
`singularity exec --nv experiment2.sif python3.10  /home/student/e21/e215736/experiment2/g-sam.py`
-「--nv」はGPU使いますっていう意味。

## logsディレクトリを作成。
-errorファイルとlogファイルが対で出力される。これでちゃんと実行されているかを確認するため。
-

## 以下のコマンドを実行。これで終わり。
`sbatch train.sbatch`
ーerrorファイルにエラーが出てきたらその都度sifファイルの追加設定を修正して正常に動かしたいファイルが実行できるまで繰り返す。
だめだったら、ベースimageからまた探しに行く。メンタル勝負です。

