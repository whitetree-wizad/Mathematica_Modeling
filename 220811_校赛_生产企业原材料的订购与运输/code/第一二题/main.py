# %%
from header import *
import pulp
from random import choice

# %%
df_pred = pd.read_csv('pred.csv').drop(columns='Unnamed: 0')

# %%
#获取分类数据
data = pd.read_csv('temp.csv')
index = data[data['选中'] == 1]['Unnamed: 0'].values

df_class = filter_item(data_order.loc[:,['供应商ID','材料分类']],index)

new_df_pred = pd.concat([df_class['材料分类'],df_pred],axis=1)

# %%
new_df_pred.apply(re_pre_ABC,axis=1).sum(axis=0)

# %%
#new_df_pred = new_df_pred.apply(re_pre_ABC,axis=1)

# %%
new_df_pred = new_df_pred.iloc[:,:2+24]

# %%
#预测的最大供货量的和
pred_sum = new_df_pred.iloc[:,2:].sum(axis=0)
#去除供应商id和类别后预测的最大供应量
pred_pure = new_df_pred.iloc[:,2:]
#大于2.84的索引
low_index = (pred_sum -2.84e4 < 0)
#小于2.84的索引
big_index = (pred_sum -2.84e4 > 0)

# %%
Storage_cost = 0.1
gross_profit = 1
# 开始准备线性规划数据

big_num_index = pred_sum.index.to_list()

myProblem = pulp.LpProblem('订购数量规划',sense=pulp.LpMaximize)

D_w = pred_sum.copy()
D_w.loc[big_index] = 0
D_w.loc[low_index] = 2.84e4 - D_w[low_index]
D_w = D_w.tolist()

d_w_constraint = pred_sum.copy()
d_w_constraint.loc[low_index] = 0
d_w_constraint.loc[big_index] = d_w_constraint[big_index] - 2.84e4
d_w_constraint = d_w_constraint.tolist()

d_var = pulp.LpVariable.dicts(name='d_w',indices=big_num_index,lowBound=0)

u = pulp.LpVariable.dicts(name='u_w',indices=big_num_index,lowBound=0)
for i in range(len(big_num_index)):
    u[str(i)] = 2.84e4 + pulp.lpSum([d_var[str(j)] for j in range(i+1)]) - sum(D_w[:i+1])

#定义目标
myProblem += gross_profit * pulp.lpSum(d_var) - Storage_cost * pulp.lpSum(u)

#定义约束

for i in range(len(big_num_index)):
    myProblem += (d_var[str(i)] <= d_w_constraint[i])

for i in range(len(big_num_index)):
    myProblem += (u[str(i)] >= 2.84e4*2)

myProblem += pulp.lpSum(d_var) <= sum(D_w) + 2.84e4

#solve
myProblem.solve()

# %%
#检查输出
ls = []
for v in u.values():
    print(v.name, "=", v.value())
    ls.append(v.value())
print(sum(ls))

for v in d_var.values():
    print(v.name, "=", v.varValue)

# %%
ls = []
for v in d_var.values():
    #print(v.name, "=", v.varValue)
    ls.append(v.value())
#预测的供货量
pred_order = pd.Series(ls)
#预测的最大供货量的和的复制
pred_order_temp = pred_sum.copy()
#经过加上2.84e4的遮蔽运算，得到的是每周预计供货量
pred_order_temp.loc[big_index] = 2.84e4
pred_order = pd.Series(pred_order_temp.values + pred_order.values)

# %%
pred_order

# %%
#倒序排列后的预测最大供应量数据，包含供应商ID和类别
temp = new_df_pred.iloc[::-1,:].reset_index(drop=True)
temp_pure = temp.iloc[:,2:]
out = pd.concat([temp.iloc[:,:2],sort_and_sub(temp_pure,pred_order)],axis=1)
out.to_csv('订购方案.csv',index=None)



# %%
#获取分数
index = data[data['选中'] == 1]['Unnamed: 0'].values
Fraction = pd.read_csv('总表.csv').drop(columns='Unnamed: 0')
Fraction = filter_item(Fraction,index)
Fraction = Fraction['综合排分'].values

# %%
pd.read_csv('总表.csv').drop(columns='Unnamed: 0')

# %%

#获取偏差
index = data[data['选中'] == 1]['Unnamed: 0'].values

data_order = filter_item(data_order, index)
data_supply = filter_item(data_supply, index)

pure_race_data = ((data_supply.iloc[:,2:] - data_order.iloc[:,2:]) / data_order.iloc[:,2:])
pure_race_data = pure_race_data.fillna(0)

race_mean = pure_race_data.sum(axis=1) / (data_order.iloc[:,2:] !=0).sum(axis=1)
#取偏差绝对值
race_mean_abs = race_mean

pred_order_index = pred_order[(pred_order != 0)].index.to_list()
pred_order_pure = pred_pure.loc[:,[str(i) for i in pred_order_index]]
pred_order_lp = pred_order[pred_order_index]

# %%
# 开始准备线性规划数据

def get_week_order(week):

    pred_order_pure_one_week = pred_order_pure[str(week)].tolist()

    pred_order_lp_one_week = pred_order_lp[week].tolist()

    myProblem1 = pulp.LpProblem('订购数量分配规划',sense=pulp.LpMaximize)

    z_var = pulp.LpVariable.dicts(name='z_w_'+str(week)+'_n',indices=range(len(pred_order_pure_one_week)),lowBound=0)

    #定义目标
    myProblem1 += pulp.lpSum([Fraction[i] * z_var[i]  for i in range(len(pred_order_pure_one_week))])

    #定义约束

    #myProblem1 += pulp.lpSum([z_var[i]*(1/(1+race_mean[i])) for i in range(len(pred_order_pure_one_week))]) == 2.84e4 + pred_order_lp_one_week
    myProblem1 += pulp.lpSum([z_var[i]*((1+race_mean[i])) for i in range(len(pred_order_pure_one_week))]) ==pred_order_lp_one_week

    #print(pred_order_lp_one_week)

    for i in range(len(pred_order_pure_one_week)):
        myProblem1 += z_var[i]*(1/(1+race_mean[i])) <= pred_order_pure_one_week[i]

    myProblem1.solve()

    ls = []
    i=0
    for v in z_var.values():
        #print(v.name, "=", v.varValue,'pred_order = ',pred_order_pure_one_week[i],'race =',race_mean[i])
        ls.append(v.value())
        i += 1

    #print(pulp.lpSum([z_var[i]*((1+race_mean[i])) for i in range(len(pred_order_pure_one_week))]).value() )

    out = pd.Series(ls,name=f'W{week}')

    return out

# %%
out = []
for i in range(len(big_num_index)):
    if i in pred_order_index:
        out.append(get_week_order(i))
    else:
        out.append(pred_pure[str(i)].rename(f'W{i}'))

# %%
out_order = pd.DataFrame(out).T
out_order.index = data[data['选中'] == 1]['Unnamed: 0'].values

# %%
out_order.values.sum()

# %%
(out_order.values*(race_mean.values+1).reshape(44,-1)).sum()

# %%
# 添加分类数据，方便还原
out_order.insert(0,"材料分类",new_df_pred["材料分类"].values)
out_order

# %%
#temp = (out_order.values*(race_mean.values+1).reshape(44,-1))
#pd.DataFrame(temp)

# %%
# 先还原，后导出
out_order.apply(lambda x:re_pre_ABC(x,data_index=1),axis=1).to_csv('order_24_week.csv')

# %%
#获取比率new_pred_pure_race_data
index = data[data['选中'] == 1]['Unnamed: 0'].values

data_order = filter_item(data_order, index)
data_supply = filter_item(data_supply, index)

pred_pure_race_data = (data_supply.iloc[:,2:] - data_order.iloc[:,2:]) / data_order.iloc[:,2:]

new_pred_pure_race_data = pred_pure_race_data.copy()

#储存到 dict中，方便调用
supplyer, week = new_pred_pure_race_data.shape
race_dict = {}
for i in range(supplyer):
    race_dict[i] = drop_nan(new_pred_pure_race_data.loc[i] + 1).tolist()

#设置函数方便抽取随机数

def get_all_race_random():
    ls = []
    for i in race_dict.values():
        ls.append(choice(i))
    return ls

# %%
race_mean.values+1

# %%
def set_plt_size(long=12,high=8):
    plt.rcParams['figure.figsize'] = (long,high)

# %%
2**16

# %%
pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1)).sum(axis=0).cumsum() + 2.84e4 - pd.Series([x*2.84e4 for x in range(1,24+1)])

# %%
pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1)).sum(axis=0).cumsum() + 2.84e4

# %%
out_order.iloc[:,1:].values * (race_mean.values+1).reshape(44,-1)

# %%
def 动态库存量(df):
    sum_temp = df.sum(axis=0).cumsum()+2.84e4
    temp = pd.Series([x*2.84e4 for x in range(1,len(sum_temp) +1)])
    return sum_temp - temp
动态库存量(pd.DataFrame(out_order.iloc[:,1:].values * (race_mean.values+1).reshape(44,-1)))

# %%
动态库存量(pd.DataFrame(out_order.iloc[:,1:].values * (race_mean.values+1).reshape(44,-1))).values

# %%


# %%
# 绘制预测供货量与实际供货量的仿真
sum_ls  = []
set_plt_size()
for ssjds in range(32):
    weeks = out_order.columns
    simulation = []
    for week in weeks:
        race_week = get_all_race_random()
        simulation.append(out_order[week] * race_week)

    out_simulation = pd.DataFrame(simulation).T

    #out_simulation.sum(axis=0).plot(alpha=0.4,color='gray',linewidth=0.05)
    out_simulation.sum(axis=0).plot(alpha=0.1,color='gray',linewidth=0.5)
    
    sum_ls.append(out_simulation.sum().sum()/24)
pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1)).sum().plot(color='r',linewidth=2)
#plt.savefig('fig/预测仿真.png',dpi=320)

# %%
max(12,1,122)

# %%
def 动态库存量(df):
    sum_temp = df.sum(axis=0)
    length = len(sum_temp)
    temp = 2.84e4
    ls = []
    for i in range(length):
        temp = max((temp + sum_temp[i] - 2.84e4) , 0)
        ls.append(temp)
    return pd.Series(ls)
sum_ls  = []
set_plt_size()
for ssjds in range(3200):
    weeks = out_order.columns
    simulation = []
    for week in weeks:
        race_week = get_all_race_random()
        simulation.append(out_order[week] * race_week)

    out_simulation = pd.DataFrame(simulation).T

    #out_simulation.sum(axis=0).plot(alpha=0.4,color='gray',linewidth=0.05)
    #动态库存量(out_simulation).plot(alpha=0.1,color='gray',linewidth=0.5)
    动态库存量(out_simulation).plot(alpha=0.4,color='gray',linewidth=0.05)
    
    sum_ls.append(out_simulation.sum().sum()/24)
动态库存量(pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1))).plot(color='r',linewidth=2)
plt.plot([0 for i in range(24)],color='blue',linewidth=1)
plt.savefig('fig/预测库存量仿真.png',dpi=320)

# %%
动态库存量(pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1)))

# %%
[0 for i in range(24)]

# %%
sum_ls  = []
set_plt_size()
for ssjds in range(320):
    weeks = out_order.columns
    simulation = []
    for week in weeks:
        race_week = get_all_race_random()
        simulation.append(out_order[week] * race_week)

    out_simulation = pd.DataFrame(simulation).T

    #out_simulation.sum(axis=0).plot(alpha=0.4,color='gray',linewidth=0.05)
    out_simulation.sum(axis=0).plot(alpha=0.1,color='gray',linewidth=0.5)
    
    sum_ls.append(out_simulation.sum().sum()/24)
pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1)).sum().plot(color='r',linewidth=2)
plt.savefig('fig/预测仿真.png',dpi=320)

# %%
(pd.DataFrame(out_order.values * (race_mean.values+1).reshape(44,-1)).sum().sum())/24

# %%
(sum(sum_ls))/320


