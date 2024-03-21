import numpy as np
import csv
import torch
import math
from tqdm import tqdm
import pandas as pd
from torchvision import models
import argparse

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        X.append(list(range(i, end_ix)))
        y.append(end_ix-1)
    return np.array(X), np.array(y)

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length, resnet=False, n_timesteps=96, hidden_size=64):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = hidden_size
        self.n_layers = 2
        self.n_timesteps = n_timesteps
        self.resnet = resnet
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)

        if resnet==True:
            # print("resnet true")
            self.last_layer = models.resnet18(pretrained=False)
            num_ftrs = self.last_layer.fc.in_features
            self.last_layer.fc = torch.nn.Linear(num_ftrs, 1)
            self.last_layer.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # print("resnet false")
            self.last_layer = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        #for name, p in self.l_lstm.named_parameters():
        #    if "weight" in name:
        #        torch.nn.init.orthogonal_(p)
        #    elif "bias" in name:
        #        torch.nn.init.constant_(p, 0)
        
    
    def init_hidden(self, batch_size, gpu=0):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        if gpu == 0:
            self.hidden = (hidden_state, cell_state)
        else:
            self.hidden = (hidden_state.cuda(), cell_state.cuda())
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        if self.resnet:
            x = lstm_out.contiguous().view(batch_size,1,self.n_timesteps,self.n_hidden)
        else:
            x = lstm_out.contiguous().view(batch_size,-1)
        return self.last_layer(x)
    
def main(args):
    print("loading data")
    p = '/data/chenang/LSTM/merged_dataframe.csv'

    data = pd.read_csv(p)
    data_g = data.groupby(['county'])
    
    n_features = args.features # this is number of parallel inputs
    n_timesteps = args.time_step # this is number of timesteps
    data = []
    X_train_sp_all = []
    X_test_sp_all = []
    y_train_sp_all = []
    y_test_sp_all = []
    for key in tqdm(data_g.groups.keys()):
        data.append(data_g.get_group(key))
        data[-1] = data[-1].to_numpy()
        data[-1] = data[-1][:,[3,0,1,2,4,6,7,8]+list(range(11,127))]
        data[-1] = data[-1].astype('float')

        for i in range(data[-1].shape[1]-1):
            if np.std(data[-1][:,i+1]) != 0:
                data[-1][:,i+1] = (data[-1][:,i+1]-np.mean(data[-1][:,i+1]))/np.std(data[-1][:,i+1])

        X, y = split_sequences(data[-1], n_timesteps)
        X_train_sp_all.append(X[:int(X.shape[0]*0.8)])
        X_test_sp_all.append(X[int(X.shape[0]*0.8+1):])
        y_train_sp_all.append(y[:int(y.shape[0]*0.8)])
        y_test_sp_all.append(y[int(y.shape[0]*0.8+1):])

    # convert dataset into input/output
    print("processing data")

    # create NN
    mv_nets = []
    optimizers = []
    print(args.resnet)
    for i in range(len(X_test_sp_all)):
        mv_nets.append(MV_LSTM(n_features,n_timesteps, resnet=args.resnet, n_timesteps = args.time_step, hidden_size = args.hidden_size))
        optimizers.append(torch.optim.Adam(mv_nets[-1].parameters(), lr=args.lr, weight_decay=args.weight_decay))
        
    criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
    mae = torch.nn.L1Loss()

    train_episodes = args.episode
    batch_size = args.batch_size

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(X_train), eta_min=0, last_epoch=-1)
    print("start training")
    for t in range(train_episodes):
        print(f"training: {t+1}/{train_episodes}")
        train_loss = []
        for c in tqdm(range(len(X_train_sp_all))):
            mv_nets[c].train()
            mv_nets[c].cuda()
            X_train_sp = X_train_sp_all[c]
            y_train_sp = y_train_sp_all[c]
            for b in range(0,len(X_train_sp),batch_size):
                inpt = data[c][X_train_sp[b:b+batch_size], 1:]
                target = data[c][y_train_sp[b:b+batch_size], 0]
                
                x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()
                y_batch = torch.tensor(target,dtype=torch.float32).cuda()
            
                mv_nets[c].init_hidden(x_batch.size(0), gpu=1)
                output = mv_nets[c](x_batch)
                loss = criterion(output.view(-1), y_batch)  
                train_loss.append(loss.item())
                loss.backward()
                optimizers[c].step()        
                optimizers[c].zero_grad()
        #scheduler.step()
        test_loss = []
        mae_loss = []
        test_loss_r0 = []
        mae_loss_r0 = []
        test_loss_c = []
        for c in tqdm(range(len(X_test_sp_all))):
            test_loss_tmp = []
            mv_nets[c].eval()
            X_test_sp = X_test_sp_all[c]
            y_test_sp = y_test_sp_all[c]
            for b in range(0,len(X_test_sp),batch_size):
                inpt = data[c][X_test_sp[b:b+batch_size], 1:]
                target = data[c][y_test_sp[b:b+batch_size], 0]
                
                x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()
                y_batch = torch.tensor(target,dtype=torch.float32).cuda()
            
                mv_nets[c].init_hidden(x_batch.size(0), gpu=1)
                output = mv_nets[c](x_batch)
                loss = criterion(output.view(-1), y_batch)
                test_loss.append(loss.item())
                test_loss_tmp.append(loss.item())
                mae_loss.append(mae(output.view(-1), y_batch).item())
                if c != 0:
                    test_loss_r0.append(loss.item())
                    mae_loss_r0.append(mae(output.view(-1), y_batch).item())

            mv_nets[c].train()
            test_loss_c.append(np.sqrt(np.mean(test_loss_tmp)))
        print(f"step: {t}, train_rmse: {math.sqrt(np.mean(train_loss))}, test_rmse: {math.sqrt(np.mean(test_loss))}, test_mae: {np.mean(mae_loss)}, test_rmse_r0: {math.sqrt(np.mean(test_loss_r0))}, test_mae_r0: {np.mean(mae_loss_r0)}")
        with open(f"/data/chenang/LSTM/results/{args.save_path}", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([t, math.sqrt(np.mean(train_loss)), math.sqrt(np.mean(test_loss)), np.mean(mae_loss), math.sqrt(np.mean(test_loss_r0)), np.mean(mae_loss_r0)])

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet", action='store_true')
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--features", type=int, default=123)
    parser.add_argument("--time_step", type=int, default=96)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--hidden_size", type=int, default=64)
    args = parser.parse_args()
    print(args)
    main(args)