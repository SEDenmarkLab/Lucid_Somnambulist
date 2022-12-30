from matplotlib.transforms import Bbox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.path import Path
from matplotlib.collections import PathCollection as pathcol
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import math
import sys
import seaborn as sns

def get_condition_components(str_):
    spl = str_.strip().split('_')
    cat = spl[2]
    solv = spl[3]
    base = spl[4]
    return cat,solv,base

def get_components(str_):
    spl = str_.strip().split('_')
    am = spl[0]
    br = spl[1]
    cat = spl[2]
    solv = spl[3]
    base = spl[4]
    return am,br,cat,solv,base

def get_handles_by_reactants(str_,handles_):
    out = []
    for k in handles_:
        # print(k.rsplit('_',3)[0])
        # print(str_)
        if k.rsplit('_',3)[0] == str_:
            out.append(k)
    return out

def code_solvbase(strs_:tuple):
    solv,base = strs_
    if base == 'a':
        if solv == '1':
            return 2 #Dioxane, moderately polar
        elif solv == '2':
            return 3 #tAmOH, most polar
        elif solv == '3':
            return 1 #Toluene, least polar
    elif base == 'b':
        if solv == '1':
            return 5 #Dioxane, moderately polar
        elif solv == '2':
            return 6 #tAmOH, most polar
        elif solv == '3':
            return 4 #Toluene, least polar
    elif base == 'c':
        if solv == '1':
            return 8 #Dioxane, moderately polar
        elif solv == '2':
            return 9 #tAmOH, most polar
        elif solv == '3':
            return 7 #Toluene, least polar
    



# handles_file = r"C:\Users\irine\Dropbox\PC\Desktop\handles_pred.csv"
# handle_df = pd.read_csv(handles_file,index_col=0)
# print(handle_df)

pred_file = r"C:\Users\irine\Dropbox\PC\Desktop\prediction_buffer_Aug-14-2022.csv" ### needs to reference database
pred_df = pd.read_csv(pred_file,index_col=None,header=0, low_memory=False).transpose()
print(pred_df)

# process_idx = []
# for idx in handle_df.index:
#     if "=" in idx:
#         new_ = idx.split('=')[-1].strip()
#         process_idx.append(new_)
#     else:
#         new_ = idx.strip()
#         process_idx.append(new_)

# handle_df.index = process_idx

# preds = pred_df.loc[handle_df.index,:]
preds = pred_df

preds.columns = [str(f+1) for f in range(len(preds.columns))]
arr = np.array(preds.values)
mean_ = arr.mean(axis=1)
stdev_ = arr.std(axis=1)
# print(mean_.shape,stdev_.shape,arr.shape)
preds['average'] = mean_
preds['stdev'] = stdev_

handles = preds.index
# print(map(get_components,handles))
unique_couplings = set([f[0]+'_'+f[1] for f in map(get_components,handles)])
print(unique_couplings)

__tst = sys.argv[1]

if __tst not in unique_couplings:
    raise Exception('Do not have predictions for this coupling in that inference file')

sub_hnd = get_handles_by_reactants(__tst,handles)

# print(sub_hnd)

preds_subset = preds.loc[sub_hnd,'average':'stdev']
# print(preds_subset)

cat_ = []
solv_ = []
base_ = []
solv_base_ = []

for k in preds_subset.index:
    cat,solv,base = get_condition_components(k)
    cat_.append(int(cat))
    solv_.append(solv)
    base_.append(base)
    solv_base_.append(code_solvbase((solv,base)))

preds_subset['catalyst'] = cat_
preds_subset['solvent'] = solv_
preds_subset['base'] = base_
preds_subset['code'] = solv_base_

# print(preds_subset)

x_ = np.arange(1,max(preds_subset['code'].values)+1)
y_ = np.arange(1,max(preds_subset['catalyst'].values)+1)
y_ = np.delete(y_,14)
# print(x_,y_)
z_pre = preds_subset[['average','catalyst','code']].sort_values(['catalyst','code'],inplace=False)

# print(z_pre)

test_ = z_pre[['code','catalyst','average']]

# print(test_)



yields = test_['average'].to_list()
z_ = []
for k in range(len(y_)): #y is catalysts
    temp = []
    for l in range(len(x_)): #x is conditions
        idx = 9*k  + l
        yld = yields[idx]
        # print(idx,yld)
        temp.append(yld)
    z_.append(temp)
# print(yields)
# print(z_)
Z = np.array(z_)
# print(Z.shape)

def get_cond_label(x_:int,pos):
    labels = [
        'K2CO3/Toluene',
        'K2CO3/Dioxane',
        'K2CO3/tAmOH',
        'NaOtBu/Toluene',
        'NaOtBu/Dioxane',
        'NaOtBu/tAmOH',
        'DBU/Toluene',
        'DBU/Dioxane',
        'DBU/tAmOH'
    ]
    # print(pos)
    # print(x_)
    return labels[pos]

def get_cat_label(x_,pos):
    # print(x_,pos)
    if type(pos) == None: print('error')
    y_ = np.arange(1,max(preds_subset['catalyst'].values)+1)
    y_ = np.delete(y_,14)
    y_ = y_.tolist()
    return str(y_[pos])

def round_z(z_,pos):
    val = np.round_(z_,decimals=0)
    val = int(val)
    return val
    

X,Y = np.meshgrid(x_,y_)

# for k in (X,Y,Z):
    # print(k.shape)

# print(X)
# print(Y)
# print(Z)


sns.set_theme(style='white')
Z_r = np.round(Z,decimals=0)
data = pd.DataFrame(Z_r,index=y_,columns=x_)
# print(data)

if sys.argv[2] == 'heatmap':
    N=512
    max_frac = np.max(Z_r)/100.0
    if max_frac >= 1.00: #This ensures that if max prediction is at (or above) 100, then we will just use the normal colormap
        cmap_ = 'viridis'
    else:
        cbar_top = int(max_frac*N)
        # print(cbar_top)
        other_frac = 1-max_frac
        viridis = cm.get_cmap('viridis',cbar_top)
        newcolors = viridis(np.linspace(0,1,cbar_top))
        # upper_frac = (100.0-np.max(Z_r))/100.0
        # num = int(np.round(upper_frac*N,decimals=0)-5)
        # print(num)
        # print(num/512)
        # print(np.max(Z_r))
        # print(N*other_frac)
        newcolors = np.append(newcolors,[np.array([78/256,76/256,43/256,0.60]) for f in range(int(N*other_frac))],axis=0)
        print(newcolors)
        cmap_ = ListedColormap(newcolors)


    ax = sns.heatmap(data=data,
        # cmap='viridis',
        cmap=cmap_,
        square=True,
        vmin=0.0,
        vmax=100.0,
        # vmax = np.max(Z_r),
        # center=np.max(Z_r)/2,
        cbar=True,
        cbar_kws={
            'shrink':0.75,
            'extend':'max',
            'extendrect':True,
            'ticks':[0,15,30,50,75,100],
            'spacing':'proportional',
            'label':'Predicted Yield',
            'location':'right',
            # 'extendfrac':(100.0-np.max(Z_r))/np.max(Z_r)
        },
        # center=50.0
        )
    # ax.yaxis.set_major_formatter(get_cat_label)
    ax.xaxis.set_major_formatter(get_cond_label)
    ax.tick_params('x',labelsize='small')
    plt.setp(ax.xaxis.get_majorticklabels(),rotation=-60,ha="left",rotation_mode="anchor")
    plt.subplots_adjust(bottom=0.25)
    plt.show()

if sys.argv[2] == 'violin':
    sns.set_theme(style="white")
    data = pd.Series(Z.flatten(),name='Predicted Yields')
    ax = sns.violinplot(x=data,scale="count",inner="point",cut=0)
    ax.set_xlim(0.0,100.0)
    plt.show()


if sys.argv[2] == '3d':
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    plt.tight_layout(h_pad=10.0,w_pad=3.0)
    fig.subplots_adjust(bottom=0.20)
    
    def init():
        ax1.plot_surface(X,Y,Z,cmap=cm.cividis,linewidth=0)
        ax1.set_zlim(0.0,100.0)
        ax1.zaxis.set_major_formatter(round_z)
        ax1.zaxis.set_major_locator(LinearLocator(10))
        ax1.yaxis.set_major_formatter(get_cat_label)
        ax1.xaxis.set_major_formatter(get_cond_label)
        ax1.yaxis.set_major_locator(LinearLocator(20))
        ax1.xaxis.set_major_locator(LinearLocator(9))
        ax1.tick_params('x',labelsize='small')
        plt.setp(ax1.xaxis.get_majorticklabels(),rotation=-35,ha="left",rotation_mode="anchor")
        plt.setp(ax1.yaxis.get_majorticklabels(),rotation=45,ha="right",rotation_mode="anchor")
        # plt.subplots_adjust(up=0.5)
        # plt.tight_layout(h_pad=10.0,w_pad=3.0)
        fig.subplots_adjust(bottom=0.15)
        return fig,    

    def anim_func(i):
        ax1.view_init(45,202-20*math.cos((math.pi*i)/180))
        return fig,

    # ani = animation.FuncAnimation(fig,anim_func,init_func=init,frames=360,blit=True)
    # ani.save(__tst+'085inc.gif',fps=25,dpi=300)
    plt.savefig(__tst+'085inc.svg')
    plt.show()

# for angle in range(0,360):
#     ax1.view_init(30,angle)
#     plt.draw()
#     plt.pause(0.5)



# preds.to_csv('processed_handles_preds_3_17_upd.csv')
