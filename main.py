import config_with_yaml as config
from opts import parser
import numpy as np
import torch
from ops.load_data import load_data
from models.UEDNet import UEDNet
from ops.dataloader import UEDNetDataset
import random 
from tqdm import tqdm

args = parser.parse_args()

def main():
    global args
    print("\n")
    print("Multi-Modal Unsupervised Encoder-Decoder Ensemble Network for Defeact Detection ")
    print("-"*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device      : ", device)
    print("Device Name : ", torch.cuda.get_device_name(0))
    print("-"*40)
    

    if args.arch =='UEDNet':
        cfg = config.load("config\\UEDNet.yml")
        print('=> DB : {}'.format(cfg.getProperty("config.data.name")))
        data,in_dim = load_data(cfg.getProperty("config.data.path"))
        print('=> UEDNet AutoEncoder loading...')
        model = UEDNet(
            in_dim = in_dim,
            hidden_size = int(cfg.getProperty("config.model.hidden_size")),
            lr = cfg.getProperty("config.model.lr")
            ).cuda()

    batch_size = 128
    epochs = 100
    Loader = UEDNetDataset(data,batch_size)

    for epoch in range(epochs):
        batch_loss = 0.0
        for X,Y in tqdm(Loader.loader):
            X = X.cuda()
            Y = Y.cuda()
            model.optimizer.zero_grad()
            y_pred = model(X).cuda()
            loss = model.criterion(y_pred, Y).cuda()
            loss.backward()
            model.optimizer.step()
            batch_loss +=loss.item()
        print("epoch {}/{}".format(epoch,epochs)+" training loss(L1) :"+str(batch_loss))


if __name__ == "__main__":
    main()