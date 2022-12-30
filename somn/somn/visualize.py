from codecs import ignore_errors
import pandas as pd
from sys import argv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
from matplotlib import cm
from matplotlib.colors import ListedColormap

def parse_data(f: str):
    """
    Pass path to dataset file and retrieve a dataframe with extracted information about each experiment for easy analysis.
    
    """
    if os.path.ispath(f):
        fpath = f
    else:
        raise Exception('Need to specify a valid path for dataset file: ',f)
    df = pd.read_csv(fpath,index_col=0,header=None)
    df.columns=['Yield']
    handles = df.index
    cats_count = {}
    for i in range(20): #Not the fastest way to do this; fix later. 1-21 w/o 15
        if i+1 == 15:
            cats_count['21']=[]
            continue
        cats_count[str(i+1)]=[]
    am_br = []
    solv_base = []
    cats_ = []
    solv_ = []
    base_ = []
    #Iterate once through index and create multiple ordered categorical references. There will be no scrambling this way. 
    for handle in handles:
        spl = handle.strip().split('_')
        base = spl[4]
        solv = spl[3]
        cat = spl[2]
        am = spl[0]
        br = spl[1]
        cats_count[cat].append(solv+base)
        am_br.append(am+'_'+br)
        solv_base.append(solv+base)
        cats_.append(cat)
        solv_.append(solv)
        base_.append(base)
    df['am_br'] = am_br
    df['Catalyst'] = cats_
    df['solvbase'] = solv_base
    df['solv'] = solv_
    df['base'] = base_
    return df


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

def get_cat_label(x_,pos,preds_subset):
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

def parse_for_heatmap(predfile):
    """
    Just parsing section of specific heatmap function: use this to loop over many heatmap generations quickly.

    Returns a dataframe with solvent/base codes to prevent repetitive looping. 
    """
    if os.path.ispath(predfile):
        pred_file = predfile
    else:
        raise Exception('Pass valid path to prediction file: ',predfile)
    preds_df = pd.read_csv(pred_file,index_col=None,header=0, low_memory=False).transpose()
    preds = preds_df
    preds.columns = [str(f+1) for f in range(len(preds.columns))]
    arr = np.array(preds.values)
    mean_ = arr.mean(axis=1)
    stdev_ = arr.std(axis=1)
    # print(mean_.shape,stdev_.shape,arr.shape)
    preds['average'] = mean_
    preds['stdev'] = stdev_
    handles = preds.index
    unique_couplings = set([f[0]+'_'+f[1] for f in map(get_components,handles)])
    for uni in unique_couplings:
        sub_hnd = get_handles_by_reactants(uni,handles)
        preds_subset = preds.loc[sub_hnd,'average':'stdev']


def get_specific_heatmap(handle,predfile,save_=False):
    """
    Accepts files of the "buffer" format (where new predictions are rows being appended as they are made, and handles are column labels).

    Returns a matplotlib axes object

    Can optionally save as .svg for visual checks in local directory
    """
    parsed = False
    if os.path.ispath(predfile):
        pred_file = predfile
    elif type(predfile) == pd.DataFrame:
        if 'code' in predfile.columns:
            parsed == True
    else:
        raise Exception('Pass valid path to prediction file: ',predfile)
    preds_df = pd.read_csv(pred_file,index_col=None,header=0, low_memory=False).transpose()
    preds = preds_df
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
    __tst = handle
    if __tst not in unique_couplings:
        raise Exception('Do not have predictions for this coupling in that inference file')
    sub_hnd = get_handles_by_reactants(__tst,handles)
    preds_subset = preds.loc[sub_hnd,'average':'stdev']
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
    x_ = np.arange(1,max(preds_subset['code'].values)+1)
    y_ = np.arange(1,max(preds_subset['catalyst'].values)+1)
    y_ = np.delete(y_,14)
    z_pre = preds_subset[['average','catalyst','code']].sort_values(['catalyst','code'],inplace=False)
    test_ = z_pre[['code','catalyst','average']]        
    yields = test_['average'].to_list()
    z_ = []
    for k in range(len(y_)): #y is catalysts
        temp = []
        for l in range(len(x_)): #x is conditions
            idx = 9*k  + l
            yld = yields[idx]
            temp.append(yld)
        z_.append(temp)
    Z = np.array(z_)
    X,Y = np.meshgrid(x_,y_)    
    Z_r = np.round(Z,decimals=0)
    data = pd.DataFrame(Z_r,index=y_,columns=x_)
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
    ax.xaxis.set_major_formatter(get_cond_label)
    ax.tick_params('x',labelsize='small')
    plt.setp(ax.xaxis.get_majorticklabels(),rotation=-60,ha="left",rotation_mode="anchor")
    plt.setp(ax.yaxis.get_majorticklabels(),rotation=0,ha='center',rotation_mode='anchor')
    plt.subplots_adjust(bottom=0.25)
    if save_ == True:
        plt.savefig(handle+'_heat.svg')
    return ax



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