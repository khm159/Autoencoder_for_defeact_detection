# [Author] - Hyungmin Kim, Ho-bum Jeon
# [Github] - https://github.com/

import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Mulfimodal Unsupervised Encoder Decoder Networks for Motor Defeact Detection")
parser.add_argument('--results', type=str, default=None)
parser.add_argument('--arch', type=str, choices=['lstm','gru','1dconv','dilatedconv','UEDNet'])
parser.add_argument('--dilate', type=int, default=1)
parser.add_argument('--dataset', type=str)
parser.add_argument('--is_shuffle', type=bool, default=True)

