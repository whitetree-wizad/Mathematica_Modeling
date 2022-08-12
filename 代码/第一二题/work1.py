# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def get_bool(data,bool):
    return data[bool]
def sort_and_plot(data):
    return data.sort_values().reset_index(drop=True).plot()

# %%
#归一化方案
def Normal(x):
    return (x-x.mean()) / x.std()
def Min_Max(x):
    return (x-x.min()) / (x.max() - x.min())

# %%
def plot_and_save(data:pd.DataFrame, path:str, size=(24.0, 16.0)):
    plt.rcParams['figure.figsize'] = size
    sort_and_plot(data)
    plt.savefig(path)
    plt.cla()

# %%
gongying = "附件1 近5年402家供应商的相关数据.xlsx"
转运商 = "附件2 近5年8家转运商的相关数据.xlsx"

# %%
data_order = pd.read_excel(gongying,'企业的订货量')
data_supply = pd.read_excel(gongying,'供应商的供货量')

# %%
test_data = data_order.copy()

# %%
data_item_num = test_data.iloc[:,2:]

# %%
test_data['订货次数']=(data_item_num>0).sum(axis=1)
test_data['订货总量'] = data_item_num.sum(axis=1)
test_data['供货总量'] = data_supply.iloc[:,2:].sum(axis=1)

# %%
data_sub = data_item_num-data_supply.iloc[:,2:]

# %%
test_data['平均供货偏差'] = (((data_sub / data_item_num).abs().fillna(0)).sum(axis=1) / test_data['订货次数'])
test_data['单次最大供应量'] = data_supply.iloc[:,2:].max(axis=1)

# %%
test_data.iloc[:,-5:]

# %%
targets = test_data.columns[-5:]

# %%
for target in targets:
    plot_and_save(test_data[target],'fig/'+target+'.png',size=(12.0,8))

# %%
data_change = test_data.iloc[:,-5:]

# %%
for i in [ '订货总量', '供货总量', '单次最大供应量']:
    data_change[i] = np.log(data_change[i].values)

# %%
data_change = data_change.apply(Min_Max)

# %%
data_change

# %%
data_temp = np.asarray(data_change[['订货次数', '订货总量', '供货总量', '平均供货偏差', '单次最大供应量']])
#计算熵值
k = -1/np.log(402)
data_log= data_temp*np.log(data_temp)
data_log = pd.DataFrame(data_log)
data_log=data_log.fillna(0)
data_log=data_log.values
ls=[]
#计算变异指数
for i in range(5):
    e_j=k*data_log.sum(axis=0)[i]
    ls.append(e_j)
temp_list =[]
for i in ls:
    temp_list.append(1-i)
#计算权重
ls=[]
#删除错误定义
#del(sum)
for i in temp_list:
    ls.append(i/sum(temp_list))


# %%
print(ls,targets)


