import config_with_yaml as config
from opts import parser
import numpy as np
import torch
from ops.load_data import load_data
from models.autoencoder import autoencoder
from ops.dataloader import UEDNetDataset
import random 
from tqdm import tqdm

args = parser.parse_args()

def main():
    global args
    print("\n")
    print("AutoEncoder Network for Motor deafect detection ")
    print("-"*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device      : ", device)
    print("Device Name : ", torch.cuda.get_device_name(0))
    print("-"*40)
    

    if args.arch =='autoencoder':
        cfg = config.load("config\\autoencoder.yml")
        print('=> DB : {}'.format(cfg.getProperty("config.data.name")))
        data,in_dim = load_data(cfg.getProperty("config.data.path"))
        print('=> AutoEncoder loading...')
        model = autoencoder(
            in_dim = in_dim,
            hidden_size = int(cfg.getProperty("config.model.hidden_size")),
            lr = cfg.getProperty("config.model.lr")
            ).cuda()

    batch_size = 16
    epochs = 200
    Loader = UEDNetDataset(data,batch_size)

    print('=> Creating Logfile...')
    log = open('./log/train_log.txt','w')

    ## (1) train step 
    print('=> Training Start...')
    for epoch in range(epochs):
        batch_loss = 0.0
        for X,Y in tqdm(Loader.loader):
            X = X.cuda()
            model.optimizer.zero_grad()
            y_pred = model(X).cuda()
            loss = model.criterion(y_pred, X).cuda()
            loss.backward()
            model.optimizer.step()
            batch_loss +=loss.item()
        out = "[epoch {}/{}]   ".format(epoch,epochs)+" training loss(L1) :{}".format(batch_loss,".3f")+" lr : "+str(model.lr)+"\n"
        print(out)
        log.write(out)
    

    ## (2) output generation step 

    output = np.zeros([2871,128000])
    losses = np.zeros([2871,])
    cursor = 0
    print('=> Testing Start...')
    loss = torch.nn.L1Loss()
    for X,Y in tqdm(Loader.loader):
        
        with torch.no_grad():
            X = X.cuda()
            Y = Y.cuda()
            y_pred = model(X).cuda()
            y_numpy = y_pred.cpu().numpy()
            
            for i,elem in enumerate(y_numpy):
                output[cursor] = elem
                losses[cursor] = loss(X[i], y_pred[i])
                cursor +=1
                
    print('=> Prediction file generation...')
    np.save('./result/output.npy',output)
    np.save('./result/loss.npy',losses)



if __name__ == "__main__":
    main()