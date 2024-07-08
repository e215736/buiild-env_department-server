import torch

print(torch.__version__)
if torch.cuda.is_available():

    print("CUDA is available. GPU is detected.")
else:

    print("CUDA is not available. GPU is not detected.")