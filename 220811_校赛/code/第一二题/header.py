from ELOL import MyLstm_reg,pd,torch,Trans,get_num,grad_clipping,clear_output,plt,np
import pickle

def pre_ABC(x,data_index = 2):
    if x['材料分类'] == "A":
        x[data_index:] =  x[data_index:] / 0.6
    if x['材料分类'] == "B":
        x[data_index:] =  x[data_index:] / 0.66
    if x['材料分类'] == "C":
        x[data_index:] =  x[data_index:] / 0.72
    return x

def re_pre_ABC(x,data_index = 2):
    if x['材料分类'] == "A":
        x[data_index:] =  x[data_index:] * 0.6
    if x['材料分类'] == "B":
        x[data_index:] =  x[data_index:] * 0.66
    if x['材料分类'] == "C":
        x[data_index:] =  x[data_index:] * 0.72
    return x

# %%
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

# %%
class PureData(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        super().__init__()
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self,index,step=None):
        return self.X[index],self.Y[index]
    def __len__(self):
        return self.len
        

# %%
def train_begin(net, data_iter, epoch, device, lr=0.01,opim_fn = torch.optim.Adam,out_num = 1000, show_pic = True):
    # 初始化网络，在初次拟合训练时使用
    ls = []
    #迁移至GPU
    net.to(device)
    #初始 chose_Y 并选择 imf
    opimter = opim_fn(net.parameters(),lr)
    
    loss = torch.nn.MSELoss()
    X,Y = next(iter(data_iter))
    
    trans = Trans()
    y_new = trans.trans_fn(Y)
    out_time = torch.log(y_new.std() * y_new.abs().mean()/(500) + 1)
    
    for i in range(epoch):
        one_temp = []
        for X,Y in data_iter:
            X_new = trans.trans_fn(X)
            Y_new = trans.trans_fn(Y)
            yhat = net(X_new.to(device))
            y1 = Y_new.to(device)
            l = loss(yhat[5:],trans_shape(y1)[5:])
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
        
        if i % 20 == 0:
            clear_output(wait=True)
            print('out_time is loss less than',out_time,'and i is',i)
            if show_pic == True:
                plt.cla()
                temp = torch.stack(ls,dim=0)
                plt.plot(get_num(temp)[-100:])
                plt.show()
                
                plt.plot(get_num(trans_shape(y1))[:],'r')
                plt.plot(get_num(yhat)[:],'b',alpha=0.4)
                plt.show()
    return ls,trans

# %%
def trans_shape(Y):
    b,s,l = Y.shape
    y = Y.reshape(b*s,-1)[:,-1].reshape(-1,1)
    return y


# 使用选定的条目筛选项目
def filter_item(data, filter_list,key='供应商ID'):
    temp_data = data.copy()
    indexs = data[key]
    indexs.name=None
    temp_data.index = indexs
    output_data = temp_data.loc[filter_list,:]
    return output_data.reset_index(drop=True)

#传入一个数据框和总和，按顺序取到所有累加值，直至等于总和
def sort_and_sub(data,total):
    def cumsum_to_sub(temp_one):
        temp_index = temp_one[1:].cumsum() > temp_one['temp']
        temp_two = temp_one[1:].copy()
        temp_two.loc[temp_index] = 0
        idx_max = len(temp_two[temp_index==False])
        if idx_max < len(temp_two):
            temp_two.loc[idx_max] = temp_one['temp'] - temp_two.sum()
        return temp_two
    temp_data = data[:]
    temp_total = total[:]
    temp_total.index = temp_data.columns
    temp_total.name='temp'
    new_temp = pd.concat([pd.DataFrame(temp_total).T,temp_data],axis=0)
    temp_temp = new_temp.apply(cumsum_to_sub)
    return temp_temp

#删除Nan
def drop_nan(serise):
    temp_serise = serise.copy()
    nan_index = (pd.isna(temp_serise)!=True)
    return temp_serise.loc[nan_index]

def set_plt_size(long=12,high=8):
    plt.rcParams['figure.figsize'] = (long,high)



def predict_product(data,pred_step = 48, length = 48 , step = 24, batchSize = 12, hidden = 256 , max_epoch = 1000, min_epoch = 150, device = 'cpu',show_pic = True):
    temp_data = np.array(data,dtype=np.float32)
    dataX = DataSeq(DataSeq(temp_data[:-1],length),step)
    dataY = DataSeq(DataSeq(temp_data[1:],length),step)
    data = PureData(dataX,dataY)
    data_iter = torch.utils.data.DataLoader(data,batchSize,shuffle=True)
    X,Y = next(iter(data_iter))

    net = MyLstm_reg(length, hidden)
    ls,trans = train_begin(net,data_iter,max_epoch,device,out_num=min_epoch,show_pic=show_pic)
    new_temp = torch.tensor(temp_data[-(length+step):])
    new_temp = trans.trans_fn(new_temp).numpy()

    for i in range(pred_step):
        dataX = DataSeq(DataSeq(new_temp[:-1],length),step)
        dataY = DataSeq(DataSeq(new_temp[1:],length),step)
        data = PureData(dataX,dataY)
        data_iter = torch.utils.data.DataLoader(data,batchSize)
        X,Y = next(iter(data_iter))
        X_new = Y
        yhat = net(X_new.to(device))
        new_temp = np.concatenate((new_temp,get_num(yhat[-1])))[-(length+step):]
        
    one = torch.tensor(new_temp)
    two = trans.re_trans_fn(one)
    return two,ls

#常数
gongying = "附件1 近5年402家供应商的相关数据.xlsx"
data_order = pd.read_excel(gongying,'企业的订货量')
data_supply = pd.read_excel(gongying,'供应商的供货量')

device = 'cuda:0'