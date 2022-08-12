# %%
#引入序列长度
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from PyEMD import EMD
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle
import os


#时间序列类
#传入数据，返回一个指定长度的
class DataSeq:
    def __init__(self, dataSet:np.array, step:int):
        self.data = dataSet
        self.step = step
        self.len = len(self.data) - self.step + 1

    def __getitem__(self,index,step=None):
        if step == None:
            step = self.step
        if isinstance(index,slice):
            return self.getkeys(index,step)
        return self.getkey(index,step)
    
    def getkey(self,index,step):
        data = self.data[index : index + step]
        assert len(data) == self.step, f'detaData out of index! length is {self.len} but index is {index}'
        return data

    def getkeys(self,indexSlice,step):
        start,stop = indexSlice.start, indexSlice.stop
        if start == None :
            start = 0
        if stop == None:
            stop = self.len
        else:
            stop = stop - 1
        ls = []
        for index in range(start,stop+1):
            ls.append(self.getkey(index,step))
        datas = np.array(ls)
        return datas
    
    def __len__(self):
        return self.len

    def __str__(self):
        return str(self.data)



class DateData(torch.utils.data.Dataset):

    def __init__(self,dataSet:np.array,length = 256, imf_num = 4-1, pre_num=1, dataSetWindows=1000):
        super().__init__()

        self.rawData = dataSet.copy()
        self.rawLen = len(self.rawData)
        step = min(self.rawLen, dataSetWindows) - length
        self.step = step
        dataSet = dataSet[-dataSetWindows:]

        #获取 imf_num数
        self.X = DataEMD(dataSet[:-pre_num],length,step,imf_num)
        self.imf_num = min(self.X.imf_num - 1, imf_num)
        self.Data = DataEMD(dataSet,length,step,self.imf_num)

        self.len = len(self.Data) - 1
        self.params = [length , self.imf_num, pre_num, dataSetWindows]

    def __getitem__(self,index):
        if isinstance(index,slice):
            assert index.stop is None or len(self.X) >= index.stop, 'detaData out of index!'
        else:
            assert self.len >= index, f'detaData out of index! length is {self.len} but index is {index}'
        return self.Data[index].astype(np.float32), self.Data[index+1].astype(np.float32)

    def __len__(self):
        return self.len
    
    def update(self,newData):
        dataSet = np.concatenate([self.rawData,newData],axis=0)
        self.__init__(dataSet,*self.params)
    
    def copy():
        pass

# %%
class DataEMD(DataSeq):
    def __init__(self, dataSet:np.array,length:int, step:int,imf_num=-1,emd = EMD()):
        self.rawData = dataSet
        data = emd(self.rawData,max_imf=imf_num).transpose(1,0)
        all_length,self.imf_num = data.shape
        super().__init__(DataSeq(data,length), step)
        self.emd = emd
    
    def update(self,data):
        pass
        

# %%
def re_EMD(data):
    batch, step, length, imf_num = data.shape
    ls = []
    for i in range(imf_num):
        chose = Chose_Y(i)
        temp_y = get_num(chose(data))
        ls.append(temp_y)
    d = np.concatenate(ls,axis=1)

    return d.sum(axis=1)


# %%
def get_num(Y):
    return Y.cpu().detach().numpy()

# %%
class Chose(torch.nn.Module):
    def __init__(self, imf):
        super().__init__()
        self.imf = imf

    def forward(self,X):
        #input = b * s * l * imf
        y = X[:,:,:,self.imf]
        O = y.transpose(1,0)
        return O

class Chose_Y(torch.nn.Module):
    def __init__(self, imf,length=1):
        super().__init__()
        self.imf = imf
        self.length = length

    def forward(self,X):
        #input = b * s * l * imf
        b,s,l,imf = X.shape
        y = X[:,:,-self.length:,self.imf]
        O = y.reshape(b*s,-1)
        return O

        
class MyLstm_reg(torch.nn.Module):
    def __init__(self,length,hidden, layer=2,out_num = 1):
        super().__init__()
        self.LSTM = torch.nn.LSTM(length,hidden,num_layers=layer)
        self.state = None
        self.linear = torch.nn.Linear(hidden,out_num)

    def forward(self,X):

        y, self.states= self.LSTM(X)

        s,b,l = y.shape
        h = y.reshape(s*b,l)
        o = self.linear(h)
        return o


class add_net(torch.nn.Module):
    def __init__(self,axis = 2):
        super().__init__()
        self.axis = axis

    def forward(self,X):
        """
        X = b * s * imf * hidden
        """
        return X.sum(axis = self.axis)

# %%
class Trans:

    # 标准化类，默认使用正态标准化
    def __init__(self, trans_fn = None, re_trans_fn = None):
        self.re_state = False
        if trans_fn == None:
            self.trans_fn = self._stand
            self.re_trans_fn = self._re_stand
        elif re_trans_fn != None:
            self.trans_fn = trans_fn
            self.re_trans_fn = re_trans_fn
        else:
            RuntimeError('没有传入恢复函数！')

    def _stand(self,data):
        if self.re_state == False:
            self.re_trans_params = [data.mean(),data.std()]
            self.re_state = True
        new_data = (data - self.re_trans_params[0]) / (self.re_trans_params[1])
        return new_data
    def _re_stand(self,data):
        temp_data = data * (self.re_trans_params[1])
        re_data = temp_data + self.re_trans_params[0]
        return re_data

    def _max_min(self, data):
        if self.re_state == False:
            self.re_trans_params = [data.min(),data.max()]
            self.re_state = True
        new_data = (data - self.re_trans_params[0]) / (self.re_trans_params[1] - self.re_trans_params[0])
        return new_data

    def _re_max_min(self,data):
        temp_data = data * (self.re_trans_params[1] - self.re_trans_params[0])
        re_data = temp_data + self.re_trans_params[0]
        return re_data

# %%
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# %%
def train_begin(net, data_iter, epoch, imf, device, lr=0.01,opim_fn = torch.optim.Adam,out_num = 1000, show_pic = True):
    # 初始化网络，在初次拟合训练时使用
    ls = []
    #迁移至GPU
    net.to(device)
    #初始 chose_Y 并选择 imf
    chose = Chose_Y(imf)
    opimter = opim_fn(net.parameters(),lr)
    
    loss = torch.nn.MSELoss()
    X,Y = next(iter(data_iter))
    
    trans = Trans()
    y = chose(Y)
    y_new = trans.trans_fn(y)
    out_time = torch.log(y_new.std() * y_new.abs().mean()/(500) + 1)
    
    for i in range(epoch):
        one_temp = []
        for X,Y in data_iter:
            X_new = trans.trans_fn(X)
            Y_new = trans.trans_fn(Y)
            yhat = net(X_new.to(device))
            y1 = chose(Y_new.to(device))
            l = loss(yhat[5:],y1[5:])
            if i> out_num and get_num(out_time) > get_num(l):
                return ls,trans
            opimter.zero_grad()
            l.backward()
            grad_clipping(net,1)
            opimter.step()
            one_temp.append(l)
            #ls.append(l)
        one_temp = torch.stack(one_temp,dim=0)
        ls.append(one_temp.max())
        
        if i % 20 == 0 and show_pic == True:
            clear_output(wait=True)
            print('out_time is loss less than',out_time,'and i is',i)
            plt.cla()
            temp = torch.stack(ls,dim=0)
            plt.plot(get_num(temp)[-100:])
            plt.show()
            
            plt.plot(get_num(y1)[:],'r')
            plt.plot(get_num(yhat)[:],'b',alpha=0.4)
            plt.show()
    return ls,trans

# %%
def get_net(imf,length = 64 ,hidden=256):
    #返回一个网络
    net = torch.nn.Sequential(
    Chose(imf),
    MyLstm_reg(length,hidden=hidden)
    )
    return net

# %%
def load_nets_and_trans(nets_path_ls, trans_path,length=64,hidden=256):
    nets = []
    imf_num = len(nets_path_ls)
    with open(trans_path,'rb') as f:
        trans_ls = pickle.load(f)
    for i in range(imf_num):
            net = get_net(i,length=length,hidden=hidden)
            net.load_state_dict(torch.load(nets_path_ls[i]))
            nets.append(net)
    return nets, trans_ls

# %%
def train_all_net(data_iter, imf_num, device, lr = 0.0005, min_epoch = 800, max_epoch = 10000, root_path='net',net_suffix='_lstm.pkl',trans_suffix= 'trans_ls.info',
                opim_fn = torch.optim.Adam,show_pic = False, length=64, hidden=256):
    nets= []
    trans_ls = []
    for i in range(imf_num):
        temp_net = get_net(i,length, hidden)
        ls, trans_one = train_begin(temp_net, data_iter, max_epoch,
                                    i,device, lr = lr, out_num = min_epoch,opim_fn=opim_fn,
                                    show_pic = show_pic)
        trans_ls.append(trans_one)
        torch.save(temp_net.state_dict(),root_path+ f'\\{i}'+net_suffix)
        net = get_net(i,length, hidden)
        net.load_state_dict(torch.load(root_path+ f'\\{i}'+net_suffix))
        nets.append(net)
    dump = pickle.dumps(trans_ls)
    with open(root_path+ '\\'+trans_suffix,'wb') as f:
        f.write(dump)
    return nets, trans_ls
    

# %%
def update_net(data_iter, nets, imf_num, device, lr = 0.0005, min_epoch = 800, max_epoch = 10000, root_path='net',net_suffix='_lstm.pkl',trans_suffix= 'trans_ls.info',
                opim_fn = torch.optim.Adam, show_pic = False, dump_local = False, length=64, hidden=256):
    for net in nets:
        net.train()
    trans_ls = []
    for i in range(imf_num):
        ls, trans_one = train_begin(nets[i], data_iter, max_epoch, i,device,
                                    lr =  lr,out_num = min_epoch,opim_fn=opim_fn, show_pic= show_pic)
        trans_ls.append(trans_one)
        if dump_local == True:
            torch.save(nets[i].state_dict(),root_path+ f'\\{i}'+net_suffix)
            net = get_net(i, length, hidden)
            net.load_state_dict(torch.load(root_path+ f'\\{i}'+net_suffix))
            nets[i] = net

    if dump_local == True:
        dump = pickle.dumps(trans_ls)
        with open(root_path+ '\\'+trans_suffix,'wb') as f:
            f.write(dump)
    return nets, trans_ls

# %%
def predict_one(nets,trans_ls,data_iter):
    for net in nets:
        net.eval()
        net.to('cpu')
    X,Y = next(iter(data_iter))
    new_in = torch.cat([X,Y[0:1,-2:-1]],dim=1)
    imf_num = len(nets)

    pred_Y = torch.zeros_like(nets[0](new_in))

    for i in range(imf_num):
        trans_in_i = trans_ls[i].trans_fn(new_in)
        trans_Y = nets[i](trans_in_i)
        pred_Y += trans_ls[i].re_trans_fn(trans_Y)
    return pred_Y



# %%
class ELOL:
    """
    Emd LSTM OnLine Learning Module
    """
    def __init__(self,length,imf_num,hidden,rawData, device, pre_num=1, dataSetWindows=1000):

        self.length = length

        self.hidden = hidden
        self.data = DateData(rawData, length, imf_num-1, pre_num=pre_num, dataSetWindows = dataSetWindows)
        self.imf_num = self.data.imf_num
        self.data_iter = torch.utils.data.DataLoader(self.data, batch_size = 1)
        self.device = device
    
    def init_nets(self, lr = 0.0005, min_epoch = 2000, 
                    max_epoch = 10000, root_path='net',
                    net_suffix='_lstm.pkl',trans_suffix= 'trans_ls.info',
                    opim_fn = torch.optim.Adam,show_pic = True):

        try:
            os.mkdir(root_path)
        except:
            print(f'文件夹 {root_path} 已经存在……开始训练网络')
        self.nets, self.trans_ls = train_all_net(self.data_iter,self.imf_num + 1, self.device,
                                                lr = lr, min_epoch = min_epoch, 
                                                max_epoch = max_epoch, root_path = root_path,
                                                net_suffix = net_suffix,trans_suffix= trans_suffix,
                                                opim_fn = opim_fn,show_pic = show_pic, length=self.length, hidden = self.hidden)


    def load_nets_and_trans(self, nets_path, trans_path):
            self.nets, self.trans_ls = load_nets_and_trans(nets_path,trans_path,
                                                            length=self.length, hidden= self.hidden)

    
    def update_data_and_net(self,data , lr = 0.005,
                        min_epoch = 500, max_epoch = 10000,
                        root_path='net',net_suffix='_lstm.pkl',
                        trans_suffix= 'trans_ls.info',
                        opim_fn = torch.optim.Adam,
                        show_pic = False, dump_local = False):
        self.data.update(data)
        self.data_iter = torch.utils.data.DataLoader(self.data, batch_size = 1)
        self.nets, self.trans_ls = update_net(self.data_iter, self.nets, self.imf_num+1, self.device,
                                            lr=lr, min_epoch= min_epoch, max_epoch=max_epoch,
                                            root_path=root_path, net_suffix=net_suffix, trans_suffix= trans_suffix,
                                            opim_fn=opim_fn, show_pic= show_pic, dump_local=dump_local,length=self.length, hidden=self.hidden)


    def predict(self):
        self.pred_Y =  predict_one(self.nets,self.trans_ls,self.data_iter)
        return get_num(self.pred_Y[-1])

