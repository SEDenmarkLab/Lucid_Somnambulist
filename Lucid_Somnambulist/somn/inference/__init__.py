# This module contains material devoted to predicting new reaction outcomes using a model cache (from the learn module)
# together with partitions and features (from the search workflow).

# #### Need to define model dirs/paths:

# """
# From inference script:
# model_dir_real = r"modeling/Nov-19-2022-00-00_realcont10_outval_outte/out/"
# directory_real = r"modeling/Nov-18-2022out_te_samepre_OUTsampval_075inc_vt03_maincont/real"
# model_dir_rand = r"modeling/Nov-19-2022-00-00_randcont10_outval_outte/out/"
# directory_rand = r"modeling/Nov-18-2022out_te_samepre_OUTsampval_075inc_vt03_maincont/rand"


# organ_real = tf_organizer('real',partition_dir=directory_real)
# drive_real = tfDriver(organ_real)

# organ_rand = tf_organizer('rand',partition_dir=directory_rand)
# drive_rand = tfDriver(organ_rand)

# date__ = date.today().strftime("%b-%d-%Y")

# out_dir_path = r'modeling/nn_'+date__+'_codetest/out/'
# os.makedirs(out_dir_path,exist_ok=True)

# metrics = []
# predictions_real = []
# predictions_rand = []

# train_val_test = {}
# train_val_test['train']=[]
# train_val_test['val']=[]
# train_val_test['test']=[]


# val_hand_f = r"validation/validation_handles.csv"

# ser = pd.read_csv(val_hand_f)
# # print(len(ser.values))
# hand = [f[0].strip() for f in ser.values.tolist()]
# # print(hand)


# x_p_df = pd.read_feather(r"modeling/Nov-21-2022valonly_desc_catemb_075inc_vt03/real/valonly_prophetic_xp.feather") #REAL descriptors
# pred_idx = x_p_df.columns.tolist()
# x_p_re = assemble_descriptors_from_handles(pred_idx,sub_am_dict,sub_br_dict) #real
# x_p_ra = assemble_random_descriptors_from_handles(pred_idx,rand) #rand

# with open('prediction_buffer_real'+date__+'control.csv','a') as g:
#     g.write(','.join(pred_idx)+'\n')

# with open('prediction_buffer_rand'+date__+'control.csv','a') as g:
#     g.write(','.join(pred_idx)+'\n')

# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # try:
# #     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=46000)])
# # except RuntimeError as e:
# #     print(e)
# with tf.device('/GPU:0'):
#     for __k in range(len(drive_real.organizer.partitions)):
#         ### Feature stuff ###
#         xtr,xval,xte,ytr,yval,yte = [f[0] for f in drive_real.x_y]
#         name_ = str(drive_real.current_part_id)
#         print(name_)
#         partition_index = organ_real.partIDs.index(int(name_))
#         tr,va,te,y1,y2,y3 = drive_real.organizer.partitions[partition_index]
#         tr,va,te = drive_real._feather_to_np((tr,va,te))
#         # print(tr[0].shape,x_p_df.shape)
#         x_tr = assemble_descriptors_from_handles(tr[1].to_list(),sub_am_dict,sub_br_dict)
#         x_va = assemble_descriptors_from_handles(va[1].to_list(),sub_am_dict,sub_br_dict)
#         x_te = assemble_descriptors_from_handles(te[1].to_list(),sub_am_dict,sub_br_dict)
#         # (x_tr_,x_va_,x_te_,x_p_) = preprocess_feature_arrays((x_tr,x_va,x_te,x_p),save_mask=False)
#         x_tr_ra = assemble_random_descriptors_from_handles(tr[1].to_list(),rand)
#         x_va_ra = assemble_random_descriptors_from_handles(va[1].to_list(),rand)
#         x_te_ra = assemble_random_descriptors_from_handles(te[1].to_list(),rand)
#         (x_tr_,x_va_,x_te_,x_p_),(x_tr_ra_,x_va_ra_,x_te_ra_,x_p_ra_) = new_mask_random_feature_arrays((x_tr,x_va,x_te,x_p_re),(x_tr_ra,x_va_ra,x_te_ra,x_p_ra),_vt=1e-3)

#         ### Real ###
#         train_val_test['train']=[]
#         train_val_test['val']=[]
#         train_val_test['test']=[]
#         models__ = glob(model_dir_real+name_+"hpset*.h5")
#         # print(models__)
#         ytr = ytr.ravel()
#         yval = yval.ravel()
#         yte= yte.ravel()
#         train_val_test['train'].append(ytr)
#         train_val_test['val'].append(yval)
#         train_val_test['test'].append(yte)
#         for _model in models__:
#             ### For new inferences
#             model_config = keras.models.load_model(_model).get_config()
#             # print(model_config)
#             model_config['layers'][0]['config']['batch_input_shape'] = (None,x_tr_.transpose().shape[1]) #Update shape to match NEW preprocessing
#             model = tf.keras.Sequential.from_config(model_config)
#             model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1),
#                     loss = 'mse',
#                     metrics=['accuracy','mean_absolute_error','mean_squared_error'],
#             )
#             history = model.fit(
#                 x_tr_.transpose().to_numpy(),
#                 ytr,
#                 batch_size=32,
#                 epochs=175,
#                 validation_data=(x_va_.transpose().to_numpy(),yval),
#                 verbose=False,
#                 # workers=64,
#                 callbacks = [ModelCheckpoint(filepath=out_dir_path+name_+'real_best_model.h5',
#                             monitor='val_loss',
#                             save_best_only=True
#                             )]
#             )
#             model.load_weights(out_dir_path+name_+'real_best_model.h5')
#             ### For getting model values to plot
#             # model = keras.models.load_model(_model)
#             xtr = x_tr_.transpose().to_numpy()
#             xval = x_va_.transpose().to_numpy()
#             xte = x_te_.transpose().to_numpy()
#             x_p = x_p_.transpose().to_numpy() #Important to get index-instance column-features
#             ytr_p,yval_p,yte_p,yp_p = model_inference(model,xtr,(xval,xte,x_p))
#             print(yp_p.shape,'real')
#             predictions_real.append(yp_p)
#             train_val_test['train'].append(ytr_p)
#             train_val_test['val'].append(yval_p)
#             train_val_test['test'].append(yte_p)
#             mae_tr = mean_absolute_error(ytr,ytr_p)
#             mae_te = mean_absolute_error(yte,yte_p)
#             mae_val = mean_absolute_error(yval,yval_p)
#             metrics.append([mae_tr,mae_val,mae_te])
#             with open('prediction_buffer_real'+date__+'control.csv','a') as g:
#                 g.write(','.join([str(f) for f in yp_p])+'\n')
#         outdf = pd.DataFrame(train_val_test['train']).transpose()
#         outdf.index = x_tr_.columns
#         valdf = pd.DataFrame(train_val_test['val']).transpose()
#         valdf.index = x_va_.columns
#         tedf = pd.DataFrame(train_val_test['test']).transpose()
#         tedf.index = x_te_.columns
#         outdf_ = pd.concat((outdf,valdf,tedf),axis=0,keys=['train','val','test'])
#         outdf_.to_csv(out_dir_path+'output_real_models_part'+name_+'_'+'_'.join([str(len(ytr)),
#                         str(len(yval)),str(len(yte))])+'.csv')

#         ### Rand ###
#         train_val_test['train']=[] #clear cache
#         train_val_test['val']=[]
#         train_val_test['test']=[]
#         models__ = glob(model_dir_rand+name_+"hpset*.h5") #get random feature models
#         print(models__)
#         ytr = ytr.ravel() #Y is same between random and real - these are made at the same time
#         yval = yval.ravel()
#         yte= yte.ravel()
#         train_val_test['train'].append(ytr)
#         train_val_test['val'].append(yval)
#         train_val_test['test'].append(yte)
#         for _model in models__:
#             ### For new inferences
#             model_config = keras.models.load_model(_model).get_config()
#             # print(model_config)
#             model_config['layers'][0]['config']['batch_input_shape'] = (None,x_tr_.transpose().shape[1]) #random shape SHOULD be the same; if it is not, something is wrong
#             model = tf.keras.Sequential.from_config(model_config)
#             model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1),
#                     loss = 'mse',
#                     metrics=['accuracy','mean_absolute_error','mean_squared_error'],
#             )
#             history = model.fit(
#                 x_tr_ra_.transpose().to_numpy(), #This is fit on the random features
#                 ytr,
#                 batch_size=32,
#                 epochs=175,
#                 validation_data=(x_va_ra_.transpose().to_numpy(),yval), #This is validated on random features
#                 verbose=False,
#                 callbacks = [ModelCheckpoint(filepath=out_dir_path+name_+'rand_best_model.h5',
#                             monitor='val_loss',
#                             save_best_only=True
#                             )]
#             )
#             model.load_weights(out_dir_path+name_+'rand_best_model.h5')
#             ### For getting model values to plot
#             # model = keras.models.load_model(_model)
#             ## Prep random features for inferences to get final values ##
#             xtr = x_tr_ra_.transpose().to_numpy()
#             xval = x_va_ra_.transpose().to_numpy()
#             xte = x_te_ra_.transpose().to_numpy()
#             x_p = x_p_ra_.transpose().to_numpy() #Important to get index-instance column-features
#             ytr_p,yval_p,yte_p,yp_p = model_inference(model,xtr,(xval,xte,x_p))
#             print(yp_p.shape,'rand')
#             predictions_rand.append(yp_p)
#             train_val_test['train'].append(ytr_p)
#             train_val_test['val'].append(yval_p)
#             train_val_test['test'].append(yte_p)
#             mae_tr = mean_absolute_error(ytr,ytr_p)
#             mae_te = mean_absolute_error(yte,yte_p)
#             mae_val = mean_absolute_error(yval,yval_p)
#             metrics.append([mae_tr,mae_val,mae_te])
#             with open('prediction_buffer_rand'+date__+'control.csv','a') as g:
#                 g.write(','.join([str(f) for f in yp_p])+'\n')
#         outdf = pd.DataFrame(train_val_test['train']).transpose()
#         outdf.index = x_tr_ra_.columns
#         valdf = pd.DataFrame(train_val_test['val']).transpose()
#         valdf.index = x_va_ra_.columns
#         tedf = pd.DataFrame(train_val_test['test']).transpose()
#         tedf.index = x_te_ra_.columns
#         outdf_ = pd.concat((outdf,valdf,tedf),axis=0,keys=['train','val','test'])
#         outdf_.to_csv(out_dir_path+'output_rand_models_part'+name_+'_'+'_'.join([str(len(ytr)),
#                         str(len(yval)),str(len(yte))])+'.csv')
#         drive_real.get_next_part()
#         print('completed part number: '+str(__k))


# pred_df_real = pd.DataFrame(predictions_real,columns=pred_idx).transpose()
# pred_df_real.to_csv(out_dir_path+'prophetic_output_real'+date__+'.csv')
# pred_df_rand = pd.DataFrame(predictions_rand,columns=pred_idx).transpose()
# pred_df_rand.to_csv(out_dir_path+'prophetic_output_rand'+date__+'.csv')


# preproc ref for val:

# def assemble_descriptors_from_handles(handle_input,am_dict,br_dict):

#     General utility for assembling ordered descriptors based on input reaction handles and
#     calculated amine and bromide rdf descriptor dictionaries. This can be used to automate
#     testing hypertuning of rdf calculator hyperparams.


#     use sysargv[1] for handle input

#     sys.argv[1] should be list of truncated handles:
#     amine_bromide,amine_bromide,....

#     OR

#     pass a list of ALL handles:
#     amine_br_cat_solv_base

#     This will assemble only descriptors as required by the list of handles, and will
#     return the descriptors in the appropriate order

#     Can also be all handles from a datafile; whatever.

#     This is meant to use am_dict and br_dict as conformer-averaged descriptors.
#     This lets the user apply different parameters to descriptor tabulation flexibly.

#     if type(handle_input) == str:
#         rxn_hndls = [f for f in handle_input.split(',') if f!='']
#         prophetic=True
#     elif type(handle_input) == list:
#         rxn_hndls = [tuple(f.rsplit('_')) for f in handle_input]
#         prophetic=False
#     else:
#         raise ValueError('Must pass manual string input of handles OR list from dataset')

#     # print(handle_input)
#     # print(rxn_hndls)
#     # outfile_name = date_+'_desc_input'
#     directory = 'descriptors/'
#     basefile = directory+'base_params.csv'
#     basedf = pd.read_csv(basefile,header=None,index_col=0).transpose()
#     solvfile = directory+'solvent_params.csv'
#     solvdf = pd.read_csv(solvfile,header=None,index_col=0).transpose()
#     # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ##Normal ASO/AEIF cats CHANGED TEST
#     catfile = '/home/nir2/tfwork/ROCHE_ws/descriptors/iso_catalyst_embedding.csv' ##isomap embedded cats CHANGED FOR SIMPLIFICATION
#     catdf = pd.read_csv(catfile,header=None,index_col=0).transpose()

#     ### Trying to assemble descriptors for labelled examples with specific conditions ###
#     if prophetic==False:
#         columns = []
#         labels=[]
#         for i,handle in enumerate(rxn_hndls):
#             am,br,cat,solv,base = handle
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             handlestring = handle_input[i]
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handlestring)
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         return outdf

#     ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
#     elif prophetic == True:
#         solv_base_cond = ['1_a','1_b','1_c','2_a','2_b','2_c','3_a','3_b','3_c']
#         allcats = [str(f+1) for f in range(21) if f != 14]
#         s = {}_{}_{}
#         exp_handles = []
#         for combination in itertools.product(rxn_hndls,allcats,solv_base_cond):
#             exp_handles.append(s.format(*combination))
#         columns = []
#         labels=[]
#         for handle in exp_handles:
#             am,br,cat,solv,base = tuple(handle.split('_'))
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handle)
#             # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf


# def preprocess_feature_arrays(feature_dataframes: (pd.DataFrame),labels: list = None,save_mask = False,_vt=None):

#     NOTE: labels depreciated until further development

#     Accepts tuple of dataframes with raw descriptors, then preprocesses them.

#     Outputs them as a combined df with labels to retrieve them from labels parameter.
#     This ensures equal preprocessing across each feature set.

#     Note: pass with COLUMNS as instances and INDICES as features, eg. df[handle]=pd.series([feat1,feat2,feat3...featn])

#     Use:
#     tuple of dfs: (train_features,validation_features,test_features,prophetic_features)
#     optional: list of names: ['train','validate','test','predict']

#     returns dfs like this:
#     tuple(traindf,validatedf,testdf,predictdf) corresponding to labels

#     OR if labels are explicitly passed, then get a df with keys as labels

#     Standard use:
#     train,val,test,pred = preprocess_feature_arrays((train_pre,val_pre,te_pre_pred_pre))

#     TO UNPACK DATAFRAME OUTPUT WHEN LABELS ARE EXPLICIT:
#     use dfout[key] to retrieve column-instance/row-feature sub dataframes

#     if labels==None:
#         labels = [str(f) for f in range(len(feature_dataframes))]
#         combined_df = pd.concat(feature_dataframes,axis=1,keys=labels) #concatenate instances on columns
#         # print(combined_df)
#         mask = list(combined_df.nunique(axis=1)!=1) # Boolean for rows with more than one unique value
#         # print(len(mask))
#         filtered_df = combined_df.iloc[mask,:] # Get only indices with more than one unique value
#         # print(filtered_df)
#         ### IAN CHANGE ADDED VARIANCE THRESHOLD ### - this was probably a mistake and may remove too many features. Scaling first is probably the correct thing to do.
#         if type(_vt) == float:
#             vt = VarianceThreshold(threshold=_vt)
#         elif _vt == 'old':
#             vt = VarianceThreshold(threshold=0.04)
#         elif _vt == None:
#             vt = VarianceThreshold(threshold=1e-4)
#         # filtered_df_scale = pd.DataFrame(np.transpose(MinMaxScaler().fit_transform(VarianceThreshold(threshold=0.04).fit_transform(filtered_df.transpose().to_numpy()))),columns=filtered_df.columns) ## No variance threshold is better for the new RDFs
#         # output = tuple([filtered_df_scale[lbl] for lbl in labels])
#         # if save_mask==True: return output,mask,filtered_df.transpose().columns,None
#         # elif save_mask==False: return output

#         # sc = MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy())
#         vt_f = vt.fit_transform(filtered_df.transpose().to_numpy())
#         sc =  MinMaxScaler().fit_transform(vt_f)
#         filtered_df_scale = pd.DataFrame(np.transpose(sc),columns=filtered_df.columns)
#         # filtered_df_scale = pd.DataFrame(np.transpose(VarianceThreshold(threshold=0.08).fit_transform(MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy()))),columns=filtered_df.columns)
#         # filtered_df_scale = pd.DataFrame(np.transpose(MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy())),columns=filtered_df.columns) ## No variance threshold is better for the new RDFs
#         # print(filtered_df_scale)
#         output = tuple([filtered_df_scale[lbl] for lbl in labels])
#         if save_mask == True:
#             return output,mask
#         elif save_mask == False:
#             return output
#     ### Too much development; following is depreciated
#     # elif len(labels) != len(feature_dataframes):
#     #     raise Exception('Must pass equal number of df labels as feature dfs')
#     # else:
#     #     combined_df = pd.concat(feature_dataframes,axis=0,keys=labels).transpose() #Gets features to columns
#     #     mask = list(combined_df.nunique(axis=0)!=1) # Boolean for columns with more than one unique value
#     #     filtered_df = combined_df.iloc[:,mask] # Get only columns with more than one unique value
#     #     if save_mask == True:
#     #         return filtered_df,mask
#     #     elif save_mask == False:
#     #         return filtered_df


# prophetic:
# def prophetic_mask_random_feature_arrays(real_feature_dataframe,rand_feat_dataframe,corr_cut=0.95,_vt=None):

#     Use preprocessing on real features to mask randomized feature arrays, creating an actual randomized feature test which
#     has proper component-wise randomization instead of instance-wise randomization, and preserves the actual input shapes
#     used for the real features.

#     rand out then real out as two dfs


#     # labels = [str(f) for f in range(len(real_feature_dataframes))]
#     # combined_df = pd.concat(real_feature_dataframes,axis=1,keys=labels) #concatenate instances on columns
#     # comb_rand = pd.concat(rand_feat_dataframes,axis=1,keys=labels)
#     mask = list(real_feature_dataframe.nunique(axis=1)!=1) # Boolean for rows with more than one unique value
#     filtered_df = real_feature_dataframe.iloc[mask,:] # Get only indices with more than one unique value
#     filtered_rand = rand_feat_dataframe.iloc[mask,:]
#     if _vt == 'old': _vt = 0.04 #This preserves an old version of vt, and the next condition ought to still be "if" so that it still runs when this is true
#     elif _vt == None: _vt = 1e-4
#     if type(_vt) == float: #Found that vt HAS to come first, or else the wrong features are removed.
#         vt = VarianceThreshold(threshold=_vt)
#         sc = MinMaxScaler()
#         vt_real = vt.fit_transform(filtered_df.transpose().to_numpy())
#         vt_rand = vt.transform(filtered_rand.transpose().to_numpy())
#         sc_vt_real = sc.fit_transform(vt_real)
#         sc_vt_rand = sc.transform(vt_rand)
#         # sc_df_real = sc.transform(filtered_df.transpose().to_numpy())
#         # sc_df_rand = sc.transform(filtered_rand.transpose().to_numpy())
#         # vt.fit(sc_df_real)
#         vt_df_real = pd.DataFrame(sc_vt_real)
#         vt_df_rand = pd.DataFrame(sc_vt_rand)
#         ### Below, replace transposed data with noc_[type] dataframes if using correlation cutoff
#         processed_rand_feats = pd.DataFrame(np.transpose(vt_df_rand.to_numpy()),columns=filtered_df.columns) #Ensures labels stripped; gives transposed arrays (row = feature, column= instance)
#         processed_real_feats = pd.DataFrame(np.transpose(vt_df_real.to_numpy()),columns=filtered_df.columns)
#         # output_rand = tuple([processed_rand_feats[lbl] for lbl in labels])
#         # output_real = tuple([processed_real_feats[lbl] for lbl in labels])
#         return processed_rand_feats,processed_real_feats


# today = date.today()
# date_ = today.strftime("%b-%d-%Y")+'valonly_desc'
# outdir = 'modeling/'+date_+"_catemb_075inc_vt03/"
# os.makedirs(outdir+'rand/',exist_ok=True)
# os.makedirs(outdir+'real/',exist_ok=True)
# name_ = 'valonly_prophetic'
# #Random features made on a component-basis
# #Random feature arrays
# x_p = assemble_random_descriptors_from_handles(hand,rand)
# #Real features used to generate masks for random features
# x_p_real = assemble_descriptors_from_handles(hand,sub_am_dict,sub_br_dict)
# (x_p_),(x_p_re) = prophetic_mask_random_feature_arrays(x_p_real,x_p,_vt=0.03)    #Comment out to only do train/test
# #These have been preprocessed just like the real features, but are randomized on a component-basis.
# x_p_.to_feather(outdir+'rand/'+name_+'_xp.feather')
# # x_va_.to_feather(outdir+'rand/'+name_+'_xva.feather')  #Comment out to only do train/test
# # x_te_.to_feather(outdir+'rand/'+name_+'_xte.feather')
# #Save these in case we want direct comparison later
# x_p_re.to_feather(outdir+'real/'+name_+'_xp.feather')
# # x_va_re.to_feather(outdir+'real/'+name_+'_xva.feather')    #Comment out to only do train/test
# # x_te_re.to_feather(outdir+'real/'+name_+'_xte.feather')
# # print(tr,tr.transpose().reset_index(drop=True))
# ### Drop index for data because it is outside of the index/col level. This fails with .to_feather()
# # tr.transpose().reset_index(drop=True).to_feather(outdir+'rand/'+name_+'_ytr.feather')
# # va.transpose().reset_index(drop=True).to_feather(outdir+'rand/'+name_+'_yva.feather')    #Comment out to only do train/test
# # te.transpose().reset_index(drop=True).to_feather(outdir+'rand/'+name_+'_yte.feather')

# # tr.transpose().reset_index(drop=True).to_feather(outdir+'real/'+name_+'_ytr.feather')
# # va.transpose().reset_index(drop=True).to_feather(outdir+'real/'+name_+'_yva.feather')    #Comment out to only do train/test
# # te.transpose().reset_index(drop=True).to_feather(outdir+'real/'+name_+'_yte.feather')


# """


# #### Separate script (this is a .py that I ran earlier in the year I think)


# import os
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# from glob import glob
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from scipy.stats import linregress
# from sklearn.metrics import mean_absolute_error
# from prototype_roche_nn_screen import tf_organizer,tfDriver
# import pickle
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# import molli as ml
# from math import sqrt
# from rdkit import Chem
# from rdkit.Chem import rdqueries
# from openbabel import openbabel
# from datetime import date
# import itertools
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import VarianceThreshold
# import random
# from mytools import corrX_new


# def get_mae_metrics(model,X: np.array,inference_x:(np.array),y: np.array,infer_labels:(np.array)):
#     """
#     Get inferences on partitions, then evaluate errors and report mae

#     2/26/2022 VAL ONLY

#     """
#     yva,yte = infer_labels
#     train_errors,val_errors,test_errors = compute_residuals(model,X,inference_x,y,(yva,yte))
#     trainmae = np.mean(train_errors)
#     valmae = np.mean(val_errors)
#     testmae = np.mean(test_errors)
#     return trainmae,valmae,testmae

# def get_mse_metrics(model,X: np.array,inference_x:(np.array),y: np.array,infer_labels:(np.array)):
#     """
#     Get inferences, then evaluate mse

#     2/26/2022 VAL ONLY

#     """
#     yva,yte = infer_labels
#     train_errors,val_errors,test_errors = compute_residuals(model,X,inference_x,y,(yva,yte))
#     trainmse = np.sqrt(np.sum(np.square(train_errors)))/len(train_errors)
#     valmse = np.sqrt(np.sum(np.square(val_errors)))/len(val_errors)
#     testmse = np.sqrt(np.sum(np.square(test_errors)))/len(test_errors)
#     return trainmse,valmse,testmse


# def compute_residuals(model,X,inference_x:(np.array),y,infer_labels:(np.array)):
#     """
#     Get residual errors for partitions

#     2/26/2022 VAL ONLY

#     """
#     yva,yte = infer_labels
#     ytr_p,yva_p,yte_p = model_inference(model,X,inference_x)
#     train_errors = abs(ytr_p-y)
#     val_errors = abs(yva_p-yva)
#     test_errors = abs(yte_p-yte)
#     return train_errors,val_errors,test_errors


# def model_inference(model,X,inference_x:(np.array)):
#     """
#     Takes inference tuple, and processes it. This has val , test, and prophetic X.

#     Use trained model (or instantiated from identified parameters)

#     Outputs predicted values based on descriptors
#     """
#     if len(inference_x) == 3:
#         X_val,X_test,Xp = inference_x
#         ytr_p = model.predict(X).ravel()
#         yte_p = model.predict(X_test).ravel()
#         yva_p = model.predict(X_val).ravel()
#         ypr_p = model.predict(Xp).ravel()
#         return ytr_p,yva_p,yte_p,ypr_p
#     elif len(inference_x) == 2:
#         X_val,X_test = inference_x
#         ytr_p = model.predict(X).ravel()
#         yte_p = model.predict(X_test).ravel()
#         yva_p = model.predict(X_val).ravel()
#         return ytr_p,yva_p,yte_p
#     elif len(inference_x) == 1:
#         X_test = inference_x
#         ytr_p = model.predict(X).ravel()
#         yte_p = model.predict(X_test).ravel()
#         return ytr_p,yte_p
#     else: raise Exception('Pass inference array; did not get proper number')


# def _trans_xy_(desc: pd.DataFrame):
#     """
#     Output feature arrays from input dataframes.

#     Note: this is designed for an ecosystem which puts instances as columns and features as
#     rows.

#     Returns array and labels
#     """
#     transposition = desc.transpose()
#     if type(transposition.index[0]) != str:
#         raise ValueError('Check dataframe construction')
#     feature_array = transposition.to_numpy()
#     return feature_array,transposition.index

# def _feather_to_np(paths: tuple):
#     """
#     Take objects and output dfs of objects in same order

#     Transposes them so that instance names are now index (they are column labels in .feather storage)

#     Gives back tuple of tuples: (array, index labels) for each entry
#     """
#     out = []
#     for pth in paths:
#         df = pd.read_feather(pth)
#         df_tr = _trans_xy_(df)
#         out.append(df_tr)
#     return tuple(out)


# def plot_results(outdir: str,expkey: str,train: (np.array),test:(np.array)):
#     """
#     Plot model predicted vs observed as an image.
#     """
#     observed,predicted = test
#     # print(observed,predicted)
#     traino,trainp = train
#     fig = plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.xlim([0,100])
#     plt.ylim([0,100])
#     plt.plot(traino,trainp,'ko')
#     plt.title('Train')
#     plt.ylabel('Predicted')
#     plt.xlabel('Observed')
#     plt.text(90,3,str(mean_absolute_error(traino,trainp)))
#     k,b,r,p,_ = linregress(traino,trainp)
#     plt.plot([0,100],[0*k+b,100*k+b],alpha=0.65)
#     plt.text(75.0,7.0, f"$ \\langle R^2 \\rangle = {r**2:0.4f} $ \n $ \\langle k \\rangle = {k:0.3f} $")
#     fig.add_subplot(1,2,2)
#     plt.xlim([0,100])
#     plt.ylim([0,100])
#     plt.plot(observed,predicted,'rp')
#     plt.title('Test')
#     plt.xlabel('Observed')
#     plt.ylabel('Predicted')
#     plt.text(90,3,str(mean_absolute_error(observed,predicted)))
#     k,b,r,p,_ = linregress(observed,predicted)
#     plt.plot([0,100],[0*k+b,100*k+b],alpha=0.60)
#     plt.text(75.0,7.0, f"$ \\langle R^2 \\rangle = {r**2:0.4f} $ \n $ \\langle k \\rangle = {k:0.3f} $")
#     plt.savefig(outdir+expkey+'.png')


# def assemble_random_descriptors_from_handles(handle_input,desc:tuple):
#     """
#     Assemble descriptors from output tuple of make_randomized_features function call

#     To do this for all dataset compounds, pass every am_br joined with a comma

#     """
#     if type(handle_input) == str:
#         rxn_hndls = [f for f in handle_input.split(',') if f!='']
#         prophetic=True
#     elif type(handle_input) == list:
#         rxn_hndls = [tuple(f.rsplit('_')) for f in handle_input]
#         prophetic=False
#     else:
#         raise ValueError('Must pass manual string input of handles OR list from dataset')

#     am_dict_rand,br_dict_rand,cat_rand,solv_rand,base_rand = desc
#     basedf = base_rand
#     solvdf = solv_rand
#     catdf = cat_rand
#     br_dict = br_dict_rand
#     am_dict = am_dict_rand
#     # print(catdf)

#     ### Trying to assemble descriptors for labelled examples with specific conditions ###
#     if prophetic==False:
#         columns = []
#         labels=[]
#         for i,handle in enumerate(rxn_hndls):
#             am,br,cat,solv,base = handle
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             handlestring = handle_input[i]
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handlestring)
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf

#     ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
#     elif prophetic == True:
#         solv_base_cond = ['1_a','1_b','1_c','2_a','2_b','2_c','3_a','3_b','3_c']
#         allcats = [str(f+1) for f in range(21) if f != 14]
#         s = "{}_{}_{}"
#         exp_handles = []
#         for combination in itertools.product(rxn_hndls,allcats,solv_base_cond):
#             exp_handles.append(s.format(*combination))
#         columns = []
#         labels=[]
#         for handle in exp_handles:
#             am,br,cat,solv,base = tuple(handle.split('_'))
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handle)
#             # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf

# def assemble_descriptors_from_handles(handle_input,am_dict,br_dict):
#     """
#     General utility for assembling ordered descriptors based on input reaction handles and
#     calculated amine and bromide rdf descriptor dictionaries. This can be used to automate
#     testing hypertuning of rdf calculator hyperparams.


#     use sysargv[1] for handle input

#     sys.argv[1] should be list of truncated handles:
#     amine_bromide,amine_bromide,....

#     OR

#     pass a list of ALL handles:
#     amine_br_cat_solv_base

#     This will assemble only descriptors as required by the list of handles, and will
#     return the descriptors in the appropriate order

#     Can also be all handles from a datafile; whatever.

#     This is meant to use am_dict and br_dict as conformer-averaged descriptors.
#     This lets the user apply different parameters to descriptor tabulation flexibly.

#     """
#     if type(handle_input) == str:
#         rxn_hndls = [f for f in handle_input.split(',') if f!='']
#         prophetic=True
#     elif type(handle_input) == list:
#         rxn_hndls = [tuple(f.rsplit('_')) for f in handle_input]
#         prophetic=False
#     else:
#         raise ValueError('Must pass manual string input of handles OR list from dataset')

#     # print(handle_input)
#     # print(rxn_hndls)
#     # outfile_name = date_+'_desc_input'
#     directory = 'descriptors/'
#     basefile = directory+'base_params.csv'
#     basedf = pd.read_csv(basefile,header=None,index_col=0).transpose()
#     solvfile = directory+'solvent_params.csv'
#     solvdf = pd.read_csv(solvfile,header=None,index_col=0).transpose()
#     # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ##Normal ASO/AEIF cats CHANGED TEST
#     catfile = '/home/nir2/tfwork/ROCHE_ws/descriptors/iso_catalyst_embedding.csv' ##isomap embedded cats CHANGED FOR SIMPLIFICATION
#     catdf = pd.read_csv(catfile,header=None,index_col=0).transpose()

#     ### Trying to assemble descriptors for labelled examples with specific conditions ###
#     if prophetic==False:
#         columns = []
#         labels=[]
#         for i,handle in enumerate(rxn_hndls):
#             am,br,cat,solv,base = handle
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             handlestring = handle_input[i]
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handlestring)
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         return outdf

#     ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
#     elif prophetic == True:
#         solv_base_cond = ['1_a','1_b','1_c','2_a','2_b','2_c','3_a','3_b','3_c']
#         allcats = [str(f+1) for f in range(21) if f != 14]
#         s = "{}_{}_{}"
#         exp_handles = []
#         for combination in itertools.product(rxn_hndls,allcats,solv_base_cond):
#             exp_handles.append(s.format(*combination))
#         columns = []
#         labels=[]
#         for handle in exp_handles:
#             am,br,cat,solv,base = tuple(handle.split('_'))
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handle)
#             # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf

# def new_mask_random_feature_arrays(real_feature_dataframes: (pd.DataFrame),rand_feat_dataframes:(pd.DataFrame),corr_cut=0.95,_vt=None):
#     """
#     Use preprocessing on real features to mask randomized feature arrays, creating an actual randomized feature test which
#     has proper component-wise randomization instead of instance-wise randomization, and preserves the actual input shapes
#     used for the real features.

#     rand out then real out as two tuples

#     """
#     labels = [str(f) for f in range(len(real_feature_dataframes))]
#     combined_df = pd.concat(real_feature_dataframes,axis=1,keys=labels) #concatenate instances on columns
#     comb_rand = pd.concat(rand_feat_dataframes,axis=1,keys=labels)
#     mask = list(combined_df.nunique(axis=1)!=1) # Boolean for rows with more than one unique value
#     filtered_df = combined_df.iloc[mask,:] # Get only indices with more than one unique value
#     filtered_rand = comb_rand.iloc[mask,:]
#     if _vt == 'old': _vt = 0.04 #This preserves an old version of vt, and the next condition ought to still be "if" so that it still runs when this is true
#     elif _vt == None: _vt = 1e-4
#     if type(_vt) == float: #Found that vt HAS to come first, or else the wrong features are removed.
#         vt = VarianceThreshold(threshold=_vt)
#         sc = MinMaxScaler()
#         vt_real = vt.fit_transform(filtered_df.transpose().to_numpy())
#         vt_rand = vt.transform(filtered_rand.transpose().to_numpy())
#         sc_vt_real = sc.fit_transform(vt_real)
#         sc_vt_rand = sc.transform(vt_rand)
#         # sc_df_real = sc.transform(filtered_df.transpose().to_numpy())
#         # sc_df_rand = sc.transform(filtered_rand.transpose().to_numpy())
#         # vt.fit(sc_df_real)
#         vt_df_real = pd.DataFrame(sc_vt_real)
#         vt_df_rand = pd.DataFrame(sc_vt_rand)
#         ### This seems to get rid of important features ###
#         # nocorr_real = corrX_new(vt_df_real,cut=corr_cut) #Gives mask for columns to drop
#         # print(len(nocorr_real))
#         # noc_real = vt_df_real.drop(columns=nocorr_real) #Columns are features still
#         # noc_rand = vt_df_rand.drop(columns=nocorr_real)
#         # print(vt_df_real)
#         # print(noc_real)
#         ### Correlation not best selection metric ###
#         ### Below, replace transposed data with noc_[type] dataframes if using correlation cutoff
#         processed_rand_feats = pd.DataFrame(np.transpose(vt_df_rand.to_numpy()),columns=filtered_df.columns) #Ensures labels stripped; gives transposed arrays (row = feature, column= instance)
#         processed_real_feats = pd.DataFrame(np.transpose(vt_df_real.to_numpy()),columns=filtered_df.columns)
#         output_rand = tuple([processed_rand_feats[lbl] for lbl in labels])
#         output_real = tuple([processed_real_feats[lbl] for lbl in labels])
#         return output_rand,output_real


# ### This is for handling IO operations with dataset, partitioning, handle/label

# def zero_nonzero_rand_splits(data_df: pd.DataFrame,validation=False,n_splits: int=1,fold:int=7,yield_cutoff: int = 1):
#     """
#     Split zero/nonzero data, THEN apply random splits function

#     Get two output streams for zero and nonzero data to train classification models

#     Can set "fuzzy" yield cutoff. This is percent yield that where at or below becomes class zero.

#     """
#     zero_mask = (data_df.to_numpy() < yield_cutoff)
#     nonzero_data = data_df[~zero_mask]
#     zero_data = data_df[zero_mask]
#     if validation == False:
#         tr_z,te_z = random_splits(zero_data,n_splits=n_splits,fold=fold)
#         tr_n,te_n = random_splits(nonzero_data,n_splits=n_splits,fold=fold)
#         tr = pd.concat(tr_z,tr_n,axis=1)
#         te = pd.concat(te_z,te_n,axis=1)
#         return tr,te
#     elif validation == True:
#         tr_z,va_z,te_z = random_splits(zero_data,n_splits=n_splits,fold=fold,validation=validation)
#         tr_n,va_n,te_n = random_splits(nonzero_data,n_splits=n_splits,fold=fold,validation=validation)
#         tr = pd.concat((tr_z,tr_n),axis=1)
#         va = pd.concat((va_z,va_n),axis=1)
#         te = pd.concat((te_z,te_n),axis=1)
#         return tr,va,te
#     else: raise ValueError('validation parameter for zero/nonzero split function must be Boolean')

# def prep_for_binary_classifier(dfs: (pd.DataFrame),yield_cutoff: int = 1):
#     """
#     Prepare data for classifier by getting class labels from continuous yields
#     """
#     out = []
#     for df in dfs:
#         df = df.where(df>yield_cutoff,other=0,inplace=True) #collapse yields at or below yield cutoff to class zero
#         df = df.where(df==0,other=1,inplace=True) #collapse yields to class one
#         out.append(df)
#     return tuple(out)


# def random_splits(data_df: pd.DataFrame,validation=False,n_splits: int=1,fold:int=7):
#     """
#     Get split handles in tuple.

#     Validation boolean decides if output is (train,test) or (train,validate,test)

#     Each is a list of handles.

#     """
#     no_exp = len(data_df.index)
#     rand_arr = np.random.randint(1,high=fold+1,size=no_exp,dtype=int)
#     if validation == False:
#         train_mask = (rand_arr > 1).tolist()
#         test_mask = (rand_arr == 1).tolist()
#         mask_list = [train_mask,test_mask]
#     elif validation == True:
#         train_mask = (rand_arr > 2).tolist()
#         validate_mask = (rand_arr == 2).tolist()
#         test_mask = (rand_arr == 1).tolist()
#         mask_list = [train_mask,validate_mask,test_mask]
#     out = tuple([data_df.iloc[msk,:] for msk in mask_list])
#     return out


# def cleanup_handles(data_df: pd.DataFrame):
#     """
#     Catch-all for fixing weird typos in data entry for data files.
#     """
#     indices = data_df.index
#     strip_indices = pd.Series([f.strip() for f in indices])
#     data_df.index = strip_indices
#     # data_df.drop_duplicates(inplace=True) ## This does not work; it seems to drop most
#     data_df = data_df[~data_df.index.duplicated(keep='first')]
#     return data_df


# #### This is for handling descriptors; will be written to a class for handling that.


# def assemble_descriptors_from_handles(handle_input,am_dict,br_dict):
#     """
#     General utility for assembling ordered descriptors based on input reaction handles and
#     calculated amine and bromide rdf descriptor dictionaries. This can be used to automate
#     testing hypertuning of rdf calculator hyperparams.


#     use sysargv[1] for handle input

#     sys.argv[1] should be list of truncated handles:
#     amine_bromide,amine_bromide,....

#     OR

#     pass a list of ALL handles:
#     amine_br_cat_solv_base

#     This will assemble only descriptors as required by the list of handles, and will
#     return the descriptors in the appropriate order

#     Can also be all handles from a datafile; whatever.

#     This is meant to use am_dict and br_dict as conformer-averaged descriptors.
#     This lets the user apply different parameters to descriptor tabulation flexibly.

#     """
#     if type(handle_input) == str:
#         rxn_hndls = [f for f in handle_input.split(',') if f!='']
#         prophetic=True
#     elif type(handle_input) == list:
#         rxn_hndls = [tuple(f.rsplit('_')) for f in handle_input]
#         prophetic=False
#     else:
#         raise ValueError('Must pass manual string input of handles OR list from dataset')

#     # print(handle_input)
#     # print(rxn_hndls)
#     # outfile_name = date_+'_desc_input'
#     directory = 'descriptors/'
#     basefile = directory+'base_params.csv'
#     basedf = pd.read_csv(basefile,header=None,index_col=0).transpose()
#     solvfile = directory+'solvent_params.csv'
#     solvdf = pd.read_csv(solvfile,header=None,index_col=0).transpose()
#     # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ##Normal ASO/AEIF cats CHANGED TEST
#     catfile = '/home/nir2/tfwork/ROCHE_ws/descriptors/iso_catalyst_embedding.csv' ##isomap embedded cats CHANGED FOR SIMPLIFICATION
#     catdf = pd.read_csv(catfile,header=None,index_col=0).transpose()

#     ### Trying to assemble descriptors for labelled examples with specific conditions ###
#     if prophetic==False:
#         columns = []
#         labels=[]
#         for i,handle in enumerate(rxn_hndls):
#             am,br,cat,solv,base = handle
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             handlestring = handle_input[i]
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handlestring)
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         return outdf

#     ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
#     elif prophetic == True:
#         solv_base_cond = ['1_a','1_b','1_c','2_a','2_b','2_c','3_a','3_b','3_c']
#         allcats = [str(f+1) for f in range(21) if f != 14]
#         s = "{}_{}_{}"
#         exp_handles = []
#         for combination in itertools.product(rxn_hndls,allcats,solv_base_cond):
#             exp_handles.append(s.format(*combination))
#         columns = []
#         labels=[]
#         for handle in exp_handles:
#             am,br,cat,solv,base = tuple(handle.split('_'))
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handle)
#             # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf


# def preprocess_feature_arrays(feature_dataframes: (pd.DataFrame),labels: list = None,save_mask = False,_vt=None):
#     """
#     NOTE: labels depreciated until further development

#     Accepts tuple of dataframes with raw descriptors, then preprocesses them.

#     Outputs them as a combined df with labels to retrieve them from labels parameter.
#     This ensures equal preprocessing across each feature set.

#     Note: pass with COLUMNS as instances and INDICES as features, eg. df[handle]=pd.series([feat1,feat2,feat3...featn])

#     Use:
#     tuple of dfs: (train_features,validation_features,test_features,prophetic_features)
#     optional: list of names: ['train','validate','test','predict']

#     returns dfs like this:
#     tuple(traindf,validatedf,testdf,predictdf) corresponding to labels

#     OR if labels are explicitly passed, then get a df with keys as labels

#     Standard use:
#     train,val,test,pred = preprocess_feature_arrays((train_pre,val_pre,te_pre_pred_pre))

#     TO UNPACK DATAFRAME OUTPUT WHEN LABELS ARE EXPLICIT:
#     use dfout[key] to retrieve column-instance/row-feature sub dataframes

#     """
#     if labels==None:
#         labels = [str(f) for f in range(len(feature_dataframes))]
#         combined_df = pd.concat(feature_dataframes,axis=1,keys=labels) #concatenate instances on columns
#         # print(combined_df)
#         mask = list(combined_df.nunique(axis=1)!=1) # Boolean for rows with more than one unique value
#         # print(len(mask))
#         filtered_df = combined_df.iloc[mask,:] # Get only indices with more than one unique value
#         # print(filtered_df)
#         ### IAN CHANGE ADDED VARIANCE THRESHOLD ### - this was probably a mistake and may remove too many features. Scaling first is probably the correct thing to do.
#         if type(_vt) == float:
#             vt = VarianceThreshold(threshold=_vt)
#         elif _vt == 'old':
#             vt = VarianceThreshold(threshold=0.04)
#         elif _vt == None:
#             vt = VarianceThreshold(threshold=1e-4)
#         # filtered_df_scale = pd.DataFrame(np.transpose(MinMaxScaler().fit_transform(VarianceThreshold(threshold=0.04).fit_transform(filtered_df.transpose().to_numpy()))),columns=filtered_df.columns) ## No variance threshold is better for the new RDFs
#         # output = tuple([filtered_df_scale[lbl] for lbl in labels])
#         # if save_mask==True: return output,mask,filtered_df.transpose().columns,None
#         # elif save_mask==False: return output

#         # sc = MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy())
#         vt_f = vt.fit_transform(filtered_df.transpose().to_numpy())
#         sc =  MinMaxScaler().fit_transform(vt_f)
#         filtered_df_scale = pd.DataFrame(np.transpose(sc),columns=filtered_df.columns)
#         # filtered_df_scale = pd.DataFrame(np.transpose(VarianceThreshold(threshold=0.08).fit_transform(MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy()))),columns=filtered_df.columns)
#         # filtered_df_scale = pd.DataFrame(np.transpose(MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy())),columns=filtered_df.columns) ## No variance threshold is better for the new RDFs
#         # print(filtered_df_scale)
#         output = tuple([filtered_df_scale[lbl] for lbl in labels])
#         if save_mask == True:
#             return output,mask
#         elif save_mask == False:
#             return output
#     ### Too much development; following is depreciated
#     # elif len(labels) != len(feature_dataframes):
#     #     raise Exception('Must pass equal number of df labels as feature dfs')
#     # else:
#     #     combined_df = pd.concat(feature_dataframes,axis=0,keys=labels).transpose() #Gets features to columns
#     #     mask = list(combined_df.nunique(axis=0)!=1) # Boolean for columns with more than one unique value
#     #     filtered_df = combined_df.iloc[:,mask] # Get only columns with more than one unique value
#     #     if save_mask == True:
#     #         return filtered_df,mask
#     #     elif save_mask == False:
#     #         return filtered_df


# #### BELOW IS CODE FOR SCRAPING DICTIONARY WITH CONFORMER-SPECIFIC DESCRIPTORS AND GENERATING A PANDAS DATAFRAME OF CONFORMER-AVERAGED DESCRIPTORS
# #### THIS CODE IS WRITTEN TO ALLOW PARAMETERS TO BE TUNED SUCH AS A DISTANCE-WEIGHTING (1/R^N) AND SPHERICAL INTERVAL PARAMETER

# ### Will be written to a class smth like "rdf_calculator"

# vdw_dict = {}
# vdw_dict['C'] = 1.7
# vdw_dict['H'] = 1.2
# vdw_dict['O'] = 1.52
# vdw_dict['N'] = 1.55
# vdw_dict['F'] = 1.47
# vdw_dict['S'] = 1.8
# vdw_dict['P'] = 1.8
# vdw_dict['Cl'] = 1.75
# vdw_dict['CL'] = 1.75
# vdw_dict['Si'] = 2.1
# vdw_dict['SI'] = 2.1
# vdw_dict['Br'] = 1.85
# vdw_dict['BR'] = 1.85
# vdw_dict['I'] = 1.95

# symbols_for_rdf = [
#         'C',
#         'N',
#         'S',
#         'O',
#         'F'
#         ]

# def retrieve_bromide_rdf_descriptors(col,apd,increment: float = 1.5,radial_scale: int = 0):
#     """
#     Takes collection and json-type atom property descriptors (generated by scraping function built for xTB outputs)

#     outputs dataframe for each molecule with conformer-averaged descriptor columns and spherical slices for indices

#     These are put into a dictionary with molecule names from the collection as keys.

#     Shape of df is 20 rows (two 10 sphere slices for each half of the molecule) with 14 columns of electronic and indicator
#     RDFs

#     This should be applicable to nitrogen nucleophiles with the exception that one atom list with all atoms should be passed.
#     This would give an output with 10 extra rows that could be trimmed or just removed later with variance threshold.

#     """
#     mol_rdfs = {} #Going to store dfs in here with name for retrieval for now
#     # conf_rdfs = {}
#     # print(list(df[3]))
#     for mol in col:
#         # atom_props = apd[mol.name]
#         # print(apd.keys())
#         rdf_df = pd.DataFrame(index=['sphere_'+str(i) for i in range(10)])
#         rdf_df.name = mol.name
#         ### Get reference atoms
#         labels = [f.symbol for f in mol.atoms]
#         br_atom = mol.get_atoms_by_symbol(symbol='Br')[0]
#         br_idx = mol.atoms.index(br_atom)
#         conn = mol.get_connected_atoms(br_atom)
#         if len(conn) != 1: raise Exception('More than one group found bonded to Br atom. Check structures')
#         elif len(conn) == 1: ipso_atom = list(conn)[0]
#         else: print('foundglitch')
#         ipso_idx = mol.atoms.index(ipso_atom)
#         rdk_mol = Chem.MolFromMol2Block(mol.to_mol2(),sanitize=False)
#         if rdk_mol == None:
#             # obabel_ = ml.OpenBabelDriver(name=mol.name,scratch_dir=os.getcwd(),nprocs=1)
#             # out = obabel_.convert(mol_text=mol.to_mol2(),src="mol2",dest="smi")
#             # print(out)
#             obconv = openbabel.OBConversion()
#             obconv.SetInAndOutFormats("mol2","smi")
#             obmol = openbabel.OBMol()
#             with open('buffer.mol2','w') as g:
#                 g.write(mol.to_mol2())
#             obconv.ReadFile(obmol,'buffer.mol2')
#             obconv.Convert()
#             smi = obconv.WriteString(obmol).split()[0]
#             if '([N](=O)[O-])' in smi:
#                 smi = smi.replace('([N](=O)[O-])','([N+](=O)[O-])')
#             # print(smi)
#             rdk_mol = Chem.MolFromSmiles(smi)
#             # print(rdk_mol)
#             # break
#         leftref = get_left_reference(rdk_mol,ipso_idx,br_idx)
#         conf_rdfs = {}
#         for k,conf in enumerate(mol.conformers):
#             df = pd.DataFrame.from_dict(apd[mol.name][k],orient='columns')
#             coords = conf.coord
#             a,b,c,d = get_molplane(coords,br_idx,ipso_idx,leftref)
#             e,f,g,h = get_orthogonal_plane(coords,br_idx,ipso_idx,a,b,c,leftref)
#             h1,h2 = sort_into_halves(mol,conf,e,f,g,h)
#             for prop in df.index:
#                 rdf_ser_1 = get_rdf(coords,br_idx,h1,df.loc[prop],radial_scaling=radial_scale,inc_size=increment,first_int=1.80)
#                 rdf_ser_2 = get_rdf(coords,br_idx,h2,df.loc[prop],radial_scaling=radial_scale,inc_size=increment,first_int=1.80)
#                 if prop in conf_rdfs.keys(): conf_rdfs[prop].append([list(rdf_ser_1),list(rdf_ser_2)])
#                 else: conf_rdfs[prop]=[[list(rdf_ser_1),list(rdf_ser_2)]]
#             rdf_ser_3 = get_atom_ind_rdf(mol.atoms,coords,br_idx,h1,inc_size=increment,first_int=1.80)
#             rdf_ser_4 = get_atom_ind_rdf(mol.atoms,coords,br_idx,h2,inc_size=increment,first_int=1.80)
#         for sym,_3,_4 in zip(symbols_for_rdf,rdf_ser_3,rdf_ser_4):
#             conf_rdfs[sym+'_rdf']=[[_3,_4]]
#         desc_df = pd.DataFrame()
#         for prop,values in conf_rdfs.items():
#             array_ = np.array(values)
#             avg_array = np.mean(array_,axis=0)
#             desc_df[prop]=pd.concat([pd.Series(f) for f in avg_array],axis=0)
#         desc_df.index = ['slice_'+str(f+1) for f in range(20)]
#         mol_rdfs[mol.name]=desc_df
#     print('all done')
#     return mol_rdfs


# def get_amine_ref_n(mol: ml.Molecule):
#     """
# Returns the reference atom index for the nitrogen with an H (assumes only one)
#     """
#     nit_atm = False
#     for atm in mol.get_atoms_by_symbol(symbol='N'):
#         nbrs=mol.get_connected_atoms(atm)
#         for nbr in nbrs:
#             # print(nbr.symbol)
#             if nbr.symbol=='H':
#                 nit_atm = atm
#                 return nit_atm


# def retrieve_amine_rdf_descriptors(col,apd,increment: float = 1.1,radial_scale: int = 0):
#     """
#     Takes collection and json-type atom property descriptors (generated by scraping function built for xTB outputs)

#     outputs dataframe for each molecule with conformer-averaged descriptor columns and spherical slices for indices

#     These are put into a dictionary with molecule names from the collection as keys.

#     Shape of df is 20 rows (two 10 sphere slices for each half of the molecule) with 14 columns of electronic and indicator
#     RDFs

#     This should be applicable to nitrogen nucleophiles with the exception that one atom list with all atoms should be passed.
#     This would give an output with 10 extra rows that could be trimmed or just removed later with variance threshold.

#     """
#     mol_rdfs = {} #Going to store dfs in here with name for retrieval for now
#     # conf_rdfs = {}
#     # print(list(df[3]))
#     for mol in col:
#         # atom_props = apd[mol.name]
#         # print(apd.keys())
#         rdf_df = pd.DataFrame(index=['sphere_'+str(i) for i in range(10)])
#         rdf_df.name = mol.name
#         ### Get reference atoms
#         # labels = [f.symbol for f in mol.atoms]
#         # br_atom = mol.get_atoms_by_symbol(symbol='Br')[0]
#         n_atom = get_amine_ref_n(mol)
#         n_idx = mol.atoms.index(n_atom)
#         conn = mol.get_connected_atoms(n_atom)
#         if len(conn) == 1: raise Exception('More than one group found bonded to Br atom. Check structures')
#         # elif len(conn) == 3: ipso_atom =
#         # else: print('foundglitch')
#         # ipso_idx = mol.atoms.index(ipso_atom)
#         # rdk_mol = Chem.MolFromMol2Block(mol.to_mol2(),sanitize=False)
#         # if rdk_mol == None:
#         #     # obabel_ = ml.OpenBabelDriver(name=mol.name,scratch_dir=os.getcwd(),nprocs=1)
#         #     # out = obabel_.convert(mol_text=mol.to_mol2(),src="mol2",dest="smi")
#         #     # print(out)
#         #     obconv = openbabel.OBConversion()
#         #     obconv.SetInAndOutFormats("mol2","smi")
#         #     obmol = openbabel.OBMol()
#         #     with open('buffer.mol2','w') as g:
#         #         g.write(mol.to_mol2())
#         #     obconv.ReadFile(obmol,'buffer.mol2')
#         #     obconv.Convert()
#         #     smi = obconv.WriteString(obmol).split()[0]
#         #     if '([N](=O)[O-])' in smi:
#         #         smi = smi.replace('([N](=O)[O-])','([N+](=O)[O-])')
#         #     # print(smi)
#         #     rdk_mol = Chem.MolFromSmiles(smi)
#         #     # print(rdk_mol)
#         #     # break
#         # leftref = get_left_reference(rdk_mol,ipso_idx,n_idx)
#         conf_rdfs = {}
#         a_idx_l = [mol.atoms.index(f) for f in mol.atoms]
#         for k,conf in enumerate(mol.conformers):
#             df = pd.DataFrame.from_dict(apd[mol.name][k],orient='columns')
#             coords = conf.coord
#             # a,b,c,d = get_molplane(coords,n_idx,ipso_idx,leftref)
#             # e,f,g,h = get_orthogonal_plane(coords,n_idx,ipso_idx,a,b,c,leftref)
#             # h1,h2 = sort_into_halves(mol,conf,e,f,g,h)
#             for prop in df.index:
#                 rdf_ser_1 = get_rdf(coords,n_idx,a_idx_l,df.loc[prop],radial_scaling=radial_scale,inc_size=increment,first_int=1.20)
#                 if prop in conf_rdfs.keys(): conf_rdfs[prop].append([list(rdf_ser_1)])
#                 else: conf_rdfs[prop]=[[list(rdf_ser_1)]]
#             rdf_ser_3 = get_atom_ind_rdf(mol.atoms,coords,n_idx,a_idx_l,inc_size=increment,first_int=1.20)
#         for sym,_3 in zip(symbols_for_rdf,rdf_ser_3):
#             conf_rdfs[sym+'_rdf']=[[_3]]
#         desc_df = pd.DataFrame()
#         for prop,values in conf_rdfs.items():
#             array_ = np.array(values)
#             avg_array = np.mean(array_,axis=0)
#             desc_df[prop]=pd.concat([pd.Series(f) for f in avg_array],axis=0)
#         desc_df.index = ['slice_'+str(f+1) for f in range(10)]
#         mol_rdfs[mol.name]=desc_df
#     print('all done')
#     return mol_rdfs


# def get_rdf(coords: ml.dtypes.CartesianGeometry,reference_idx: int,atom_list,all_atoms_property_list: list,inc_size=0.90,first_int: float = 1.80,radial_scaling: int or None = 0):

#     """
#     Takes coordinates for molecule, reference atom index, list of atom indices to compute for, and property list ordered by atom idx

#     radial_scaling is an exponent for 1/(r^n) scaling the descriptors - whatever they may be

#     """
#     al = []
#     bl = []
#     cl = []
#     dl = []
#     el = []
#     fl = []
#     gl = []
#     hl = []
#     il = []
#     jl = []
#     central_atom = coords[reference_idx]
#     # print(atom_list)
#     for x in atom_list:
#         point = coords[x]
#         #print(point)
#         #print(central_atom)
#         dist = sqrt(((float(central_atom[0]) - float(point[0]))**2 + (float(central_atom[1]) - float(point[1]))**2 + (float(central_atom[2])-float(point[2]))**2))
#         # atom = ''.join([i for i in x if not i.isdigit()])
#         property = list(all_atoms_property_list)[x]
#         try:
#             property_ = float(property)
#         except:
#             property_ = 4.1888 * vdw_dict[property]**3
#         const = first_int
#         if radial_scaling==0 or radial_scaling == None: pass
#         elif type(radial_scaling) is int and radial_scaling!=0: property_ = property_ / (dist**radial_scaling)
#         else: raise ValueError('radial scaling exponent should be an integer or None')
#         if dist <= const + inc_size:                                 al.append(property_)
#         elif dist > const + inc_size and dist <= const +inc_size*2:  bl.append(property_)
#         elif dist > const +inc_size*2 and dist <= const +inc_size*3: cl.append(property_)
#         elif dist > const +inc_size*3 and dist <= const +inc_size*4: dl.append(property_)
#         elif dist > const +inc_size*4 and dist <= const +inc_size*5: el.append(property_)
#         elif dist > const +inc_size*5 and dist <= const +inc_size*6: fl.append(property_)
#         elif dist > const +inc_size*6 and dist <= const +inc_size*7: gl.append(property_)
#         elif dist > const +inc_size*7 and dist <= const +inc_size*8: hl.append(property_)
#         elif dist > const +inc_size*8 and dist <= const +inc_size*9: il.append(property_)
#         elif dist > const +inc_size*9:                               jl.append(property_)
#     series_ = pd.Series([sum(al),sum(bl),sum(cl),sum(dl),sum(el),sum(fl),sum(gl),sum(hl),sum(il),sum(jl)],
#         index = ['sphere_'+str(f+1) for f in range(10)]
#     )
#     '''
#     print al
#     print bl
#     print cl
#     print dl
#     print el
#     print fl
#     print gl
#     print hl
#     print il
#     print jl
#     '''
#     return series_

# def get_atom_ind_rdf(atoms: list[ml.dtypes.Atom],coords: ml.dtypes.CartesianGeometry,reference_idx: int,atom_list,first_int: float = 1.80,inc_size=0.90):
#     """
#     Takes atoms and returns simple binary indicator for presence of specific atom types. These are not distance-weighted.
#     """
#     atomtypes = [
#         'C',
#         'N',
#         'S',
#         'O',
#         'F'
#     ]
#     outlist = []
#     for symb in atomtypes:
#         al = []
#         bl = []
#         cl = []
#         dl = []
#         el = []
#         fl = []
#         gl = []
#         hl = []
#         il = []
#         jl = []
#         central_atom = coords[reference_idx]
#         # print(atom_list)
#         for x in atom_list:
#             point = coords[x]
#             symbol = atoms[x].symbol
#             if symbol != symb: continue
#             dist = sqrt(((float(central_atom[0]) - float(point[0]))**2 + (float(central_atom[1]) - float(point[1]))**2 + (float(central_atom[2])-float(point[2]))**2))
#             const = first_int
#             if dist <= const + inc_size:                                    al.append(1)
#             elif dist > const + inc_size and dist <= const +inc_size*2:     bl.append(1)
#             elif dist > const +inc_size*2 and dist <= const +inc_size*3:    cl.append(1)
#             elif dist > const +inc_size*3 and dist <= const +inc_size*4:    dl.append(1)
#             elif dist > const +inc_size*4 and dist <= const +inc_size*5:    el.append(1)
#             elif dist > const +inc_size*5 and dist <= const +inc_size*6:    fl.append(1)
#             elif dist > const +inc_size*6 and dist <= const +inc_size*7:    gl.append(1)
#             elif dist > const +inc_size*7 and dist <= const +inc_size*8:    hl.append(1)
#             elif dist > const +inc_size*8 and dist <= const +inc_size*9:    il.append(1)
#             elif dist > const +inc_size*9:                                  jl.append(1)
#         series_ = [sum(al),sum(bl),sum(cl),sum(dl),sum(el),sum(fl),sum(gl),sum(hl),sum(il),sum(jl)]
#         outlist.append(series_)
#     # output = np.array(outlist).T.tolist()
#     output = outlist
#     return output


# def get_molplane(coords: np.array,ref_1,ref_2,ref_3):
#     """
# Makes plane of molecule. Bromide for bromides, nitrogen for amines as ref atom. Mol is rdkit mol
#     """
#     p1 = np.array(coords[ref_1])
#     p2 = np.array(coords[ref_2])
#     p3 = np.array(coords[ref_3])
#     #print(p1,p2,p3)
#     v1 = p2-p1
#     v2 = p3-p1
#     cp = np.cross(v1,v2)
#     #print(v1)
#     #print(v2)
#     #print(cp)
#     a,b,c = cp
#     d = np.dot(cp,p1)
#     return a,b,c,d

# def get_orthogonal_plane(coords: np.array,ref_1,ref_2,a,b,c,leftref):
#     """
# Retrieve orthogonal plane to molecule, but containing reactive atom
# ref1 is the reactive atom (br or n)
# ref2 is the atom attached to it (for making a direction towards the molecule)
#     """
#     p1 = np.array(coords[ref_1])
#     p2 = np.array(coords[ref_2])
#     p4 = np.array(coords[leftref]) #for "positive" direction left/right
#     v1 = p2-p1
#     v2 = np.array([a,b,c])
#     cp = np.cross(v1,v2)
#     e,f,g = cp
#     vc = np.array([e,f,g])
#     #print('norm vect',vc)
#     #print('p1',p1)
#     #print('p4',p4)
#     #print(np.dot(vc,p4))
#     #print(leftref)
#     #print(ref1)
#     if np.dot(vc,p4) > 0:
#         h = np.dot(vc,p1)
#         return e,f,g,h
#     elif np.dot(vc,p4) < 0:
#         cp = np.cross(v2,v1)
#         e,f,g = cp
#         vc = np.array([e,f,g])
#         h = np.dot(vc,p1)
#         return e,f,g,h
#     else:
#         print('Not finding direction')


# def sort_into_halves(mol: ml.Molecule,conf: ml.dtypes.CartesianGeometry,e,f,g,h):
#     """
#     This function takes in the atom list and spits out a list of lists with atoms sorted
#     into octants. This is done with the three orthonormal planes defined by get_orthogonal_planes
#     """
#     coords: np.array = conf.coord
#     oct1 = []
#     oct2 = []
#     cp = np.array([e,f,g])
#     # cp_ = [np.float64(e),np.float64(f),np.float64(g)]
#     for i,pos in enumerate(coords):
#         direction_ = (np.tensordot(pos,cp,axes=1)-h)/abs(sqrt(e**2+f**2+g**2))
#         if direction_ > 0.15:
#                 oct1.append(i)
#         elif direction_ < -0.15:
#             oct2.append(i)
#     return [oct1,oct2]


# def get_left_reference(mol: Chem.rdchem.Mol,ipso_idx,br_idx):
#     """
#     return leftref
#     """
#     ipso_reference = mol.GetAtomWithIdx(ipso_idx)
#     br_ref = mol.GetAtomWithIdx(br_idx)
#     ortho_het,meta_het = get_ortho_meta_symbols(mol,ipso_idx)
#     # print(ortho_het,meta_het)
#     if len(ortho_het) == 0: #no ortho heteroatoms
#         less_sub = get_less_substituted_ortho(mol,ipso_idx)
#         #print(less_sub,'less_sub_ortho')
#         if less_sub == None: #ortho both the same
#             if len(meta_het) == 0: #no meta het, so using substitution
#                 less_meta_sub = get_less_substituted_meta(mol,ipso_idx)
#                 #print(less_meta_sub,'less_meta_sub')
#                 if less_meta_sub == None:
#                     nbrs = [f for f in ipso_reference.GetNeighbors() if f.GetIdx()!=br_idx]
#                     # print([f.GetIdx() for f in nbrs])
#                     leftref = nbrs[0].GetIdx() #arbitrary; symmetric
#                 elif less_meta_sub != None: #using less substituted meta atom for left reference
#                     leftref = less_meta_sub
#             elif len(meta_het) == 1: #list of tuples (symbol, idx, atomic num)
#                 leftref = meta_het[0][1]
#             elif len(meta_het) == 2:
#                 if meta_het[0][2]>meta_het[1][2]: #atomic number of first greater than atomic number of second
#                     leftref = meta_het[0][1]
#                 elif meta_het[0][2]<meta_het[1][2]:
#                     leftref = meta_het[1][1]
#                 elif meta_het[0][2]==meta_het[1][2]:
#                     leftref = meta_het[0][1] #arbitrary if they are the same
#         elif less_sub != None:
#             leftref = less_sub #If one side is less substituted AND no heteroatoms were found
#     elif len(ortho_het) == 1:
#         leftref = ortho_het[0][1] #heteroatom in ortho defines
#     elif len(ortho_het) == 2: #both ortho are het
#         if ortho_het[0][2]>ortho_het[1][2]: #atomic number of first greater than atomic number of second
#             leftref = ortho_het[0][1]
#         elif ortho_het[0][2]<ortho_het[1][2]:
#             leftref = ortho_het[1][1]
#         elif ortho_het[0][2]==ortho_het[1][2]:
#             leftref = ortho_het[0][1] #arbitrary if they are the same
#     else:
#         print('Error! Could not find bromide left reference after all conditions')
#     return leftref

# def get_ortho_meta_symbols(mol: Chem.rdchem.Mol,aryl_ref):
#     """
#     Finds out if and what heteroatoms are in the ortho-positions of aniline-type amines
#     Returns list of ortho heteroatoms and then meta heteroatoms
#     Form is: tuple (symbol,index,atomicnumber)
#     Third value can be used to sort these by importance

#     This should work for bromides!!!
#     Uses ipso carbon as reference atom.
#     """
#     pt = Chem.GetPeriodicTable()
#     # if is_aniline(mol,refn)==False:
#     #     print('error! trying to use aniline func on non aniline!')
#     #     return None
#     ar_atm = get_aromatic_atoms(mol) #all aryl atoms
#     # print(ar_atm,aryl_ref)
#     if aryl_ref not in ar_atm:
#         print('weird')
#         return None #This is weird; error here if this happens
#     het_ar_atm = [] #list of tuples describing heteroarene heteroatoms, empty if none
#     for atm in ar_atm: #Loop over aromatic atoms to find heteroaromatic atoms
#         symb = mol.GetAtomWithIdx(atm).GetSymbol()
#         if symb!='C':
#             het_ar_atm.append(tuple([symb,atm]))
#     refatom = mol.GetAtomWithIdx(aryl_ref)
#     nbrs = refatom.GetNeighbors()
#     ortho_het = []
#     meta_het = []
#     for nbr in nbrs: #This looks at ortho atoms
#         test_value = tuple([nbr.GetSymbol(),nbr.GetIdx()])
#         if test_value in het_ar_atm:
#             ortho_het.append(tuple([f for f in test_value]+[pt.GetAtomicNumber(test_value[0])]))
#         nbr2 = [f for f in nbr.GetNeighbors() if f not in nbrs]
#         for nbrr in nbr2: #This looks at one further atom out from ortho
#             test_val_2 = tuple([nbrr.GetSymbol(),nbrr.GetIdx()])
#             if test_val_2 in het_ar_atm:
#                 meta_het.append(tuple([f for f in test_val_2]+[pt.GetAtomicNumber(test_val_2[0])]))
#     # print(ortho_het,meta_het)
#     return ortho_het,meta_het

# def get_aromatic_atoms(mol: Chem.rdchem.Mol):
#     q = rdqueries.IsAromaticQueryAtom()
#     return [x.GetIdx() for x in mol.GetAtomsMatchingQuery(q)]

# def get_less_substituted_ortho(mol: Chem.rdchem.Mol,atomidx):
#     atomref = mol.GetAtomWithIdx(atomidx)
#     nbrs = atomref.GetNeighbors()
#     nbrs_ = [f for f in nbrs if f.GetSymbol()!='H' and f.GetSymbol()!='Br' and f.GetSymbol()!='N'] #No H, Br, or N
#     nbrlist = [[k.GetSymbol() for k in f.GetNeighbors()] for f in nbrs_]
#     cntlist = [f.count('H') for f in nbrlist]
#     if cntlist.count(cntlist[0]) == len(cntlist): return None #This means H count is same
#     min_v = min(cntlist)
#     min_indx = cntlist.index(min_v)
#     lesssub = nbrs_[min_indx].GetIdx()
#     return lesssub

# def get_less_substituted_meta(mol: Chem.rdchem.Mol,ipsoidx):
#     atomref = mol.GetAtomWithIdx(ipsoidx)
#     nbrs = atomref.GetNeighbors()
#     nbrs_ = [f for f in nbrs if f.GetSymbol()!='H']
#     atomrings = mol.GetRingInfo().AtomRings()
#     for ring in atomrings:
#         if ipsoidx in ring:
#             mainring = ring
#     meta_ = [[k for k in f.GetNeighbors() if k.GetIdx() not in [p.GetIdx()for p in nbrs_] and
#                                             k.GetSymbol!='H' and k.GetIdx() in mainring and
#                                             k.GetIdx()!=ipsoidx] for f in nbrs_]
#     meta_ = [p for p in meta_ if len(p)!=0]
#     # for f in meta_:
#     #     print(f,'test')
#     meta_type_list = [[k.GetSymbol() for k in f[0].GetNeighbors()] for f in meta_] #List with one item; need index in nested inner list
#     cntlist = [f.count('H') for f in meta_type_list]
#     if cntlist.count(cntlist[0]) == len(cntlist): return None #This means H count is same
#     min_v = min(cntlist)
#     min_indx = cntlist.index(min_v)
#     lesssub = meta_[min_indx][0].GetIdx()
#     #print(lesssub)
#     return lesssub

# def trim_out_of_sample(partition: tuple,reacts:str):
#     """
#     Pass a string ##_## for amine_bromide that needs to be out of sample. This function will return the partitioned dataframes with those samples
#     removed.
#     """
#     xtr,xval,xte,ytr,yval,yte = [pd.DataFrame(f[0],index=f[1]) for f in partition]
#     to_move_tr = get_handles_by_reactants(reacts,ytr.index)
#     to_move_va = get_handles_by_reactants(reacts,yval.index)
#     # to_move_tot = to_move_tr+to_move_va
#     x_trcut = xtr.loc[to_move_tr]
#     y_trcut = ytr.loc[to_move_tr]
#     xtr.drop(index=to_move_tr,inplace=True)
#     ytr.drop(index=to_move_tr,inplace=True)
#     x_vacut = xval.loc[to_move_va]
#     y_vacut = yval.loc[to_move_va]
#     xval.drop(index=to_move_va,inplace=True)
#     yval.drop(index=to_move_va,inplace=True)
#     xte = pd.concat((xte,x_trcut,x_vacut),axis=0)
#     yte = pd.concat((yte,y_trcut,y_vacut),axis=0)
#     return xtr,xval,xte,ytr,yval,yte

# def get_handles_by_reactants(str_,handles_):
#     out = []
#     for k in handles_:
#         # print(k.rsplit('_',3)[0])
#         # print(str_)
#         if k.rsplit('_',3)[0] == str_:
#             out.append(k)
#     return out

# def randomize_features(feat=np.array):
#     """
#     Accepts feature array and randomizes values

#     """
#     feat_ = feat
#     rng = np.random.default_rng()
#     feats = rng.random(out=feat)
#     return feats

# def make_randomized_features(am_dict,br_dict,catfile=None,solvfile=None,basefile=None):
#     """
#     For running randomized feature control

#     Pass dict of dataframes to this to randomize substrate features

#     Handles are the dataset partitions (as a tuple...these will be returned with the desired order but randomized)

#     output is AMINE, BROMIDE, CATALYST, SOLVENT, BASE
#     """
#     directory = 'descriptors/'

#     if basefile==None: basefile = directory+'base_params.csv'
#     else: basefile = basefile
#     basedf = pd.read_csv(basefile,header=None,index_col=0).transpose()
#     if solvfile==None: solvfile = directory+'solvent_params.csv'
#     else: solvfile==solvfile
#     solvdf = pd.read_csv(solvfile,header=None,index_col=0).transpose()
#     if catfile==None: catfile = directory+'cat_aso_aeif_combined_11_2021.csv'
#     else: catfile==catfile
#     catdf = pd.read_csv(catfile,header=None,index_col=0).transpose()
#     cat_rand = randomize_features(catdf.to_numpy())
#     catdfrand = pd.DataFrame(cat_rand,index=catdf.index,columns=catdf.columns)
#     solv_rand = randomize_features(solvdf.to_numpy())
#     solvdfrand = pd.DataFrame(solv_rand,index=solvdf.index,columns=solvdf.columns)
#     base_rand = randomize_features(basedf.to_numpy())
#     basedfrand = pd.DataFrame(base_rand,index=basedf.index,columns=basedf.columns)
#     br_dict_rand = {}
#     am_dict_rand = {}
#     for k,v in am_dict.items():
#         rand_f = randomize_features(np.array(v.iloc[:,:9].to_numpy()))
#         rand_int = np.random.randint(0,3,v.iloc[:,9:].to_numpy().shape)
#         concat = np.concatenate((rand_f,rand_int),axis=1)
#         am_dict_rand[k] = pd.DataFrame(concat,index=v.index,columns=v.columns)
#     for k,v in br_dict.items():
#         rand_f = randomize_features(np.array(v.iloc[:,:9].to_numpy()))
#         rand_int = np.random.randint(0,3,v.iloc[:,9:].to_numpy().shape)
#         concat = np.concatenate((rand_f,rand_int),axis=1)
#         br_dict_rand[k] = pd.DataFrame(concat,index=v.index,columns=v.columns)
#     return am_dict_rand,br_dict_rand,catdfrand,solvdfrand,basedfrand

# def assemble_random_descriptors_from_handles(handle_input,desc:tuple):
#     """
#     Assemble descriptors from output tuple of make_randomized_features function call

#     To do this for all dataset compounds, pass every am_br joined with a comma

#     """
#     if type(handle_input) == str:
#         rxn_hndls = [f for f in handle_input.split(',') if f!='']
#         prophetic=True
#     elif type(handle_input) == list:
#         rxn_hndls = [tuple(f.rsplit('_')) for f in handle_input]
#         prophetic=False
#     else:
#         raise ValueError('Must pass manual string input of handles OR list from dataset')

#     am_dict_rand,br_dict_rand,cat_rand,solv_rand,base_rand = desc
#     basedf = base_rand
#     solvdf = solv_rand
#     catdf = cat_rand
#     br_dict = br_dict_rand
#     am_dict = am_dict_rand
#     # print(catdf)

#     ### Trying to assemble descriptors for labelled examples with specific conditions ###
#     if prophetic==False:
#         columns = []
#         labels=[]
#         for i,handle in enumerate(rxn_hndls):
#             am,br,cat,solv,base = handle
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             handlestring = handle_input[i]
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handlestring)
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf

#     ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
#     elif prophetic == True:
#         solv_base_cond = ['1_a','1_b','1_c','2_a','2_b','2_c','3_a','3_b','3_c']
#         allcats = [str(f+1) for f in range(21) if f != 14]
#         s = "{}_{}_{}"
#         exp_handles = []
#         for combination in itertools.product(rxn_hndls,allcats,solv_base_cond):
#             exp_handles.append(s.format(*combination))
#         columns = []
#         labels=[]
#         for handle in exp_handles:
#             am,br,cat,solv,base = tuple(handle.split('_'))
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key,val in am_dict[am].iteritems(): #This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key,val in br_dict[br].iteritems():
#                 brdesc.extend(val.tolist())
#             columns.append(amdesc+brdesc+catdesc+solvdesc+basedesc)
#             labels.append(handle)
#             # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
#         outdf = pd.DataFrame(columns,index=labels).transpose()
#         # print(outdf)
#         return outdf

# def _depreciated_version_mask_random_feature_arrays(real_feature_dataframes: (pd.DataFrame),rand_feat_dataframes:(pd.DataFrame),corr_cut=0.95,_vt=None):
#     """
#     Use preprocessing on real features to mask randomized feature arrays, creating an actual randomized feature test which
#     has proper component-wise randomization instead of instance-wise randomization, and preserves the actual input shapes
#     used for the real features.

#     rand out then real out as two tuples

#     """
#     labels = [str(f) for f in range(len(real_feature_dataframes))]
#     combined_df = pd.concat(real_feature_dataframes,axis=1,keys=labels) #concatenate instances on columns
#     comb_rand = pd.concat(rand_feat_dataframes,axis=1,keys=labels)
#     mask = list(combined_df.nunique(axis=1)!=1) # Boolean for rows with more than one unique value
#     filtered_df = combined_df.iloc[mask,:] # Get only indices with more than one unique value
#     filtered_rand = comb_rand.iloc[mask,:]
#     if _vt == 'old': _vt = 0.04 #This preserves an old version of vt, and the next condition ought to still be "if" so that it still runs when this is true
#     elif _vt == None: _vt = 1e-4
#     if type(_vt) == float:
#         vt = VarianceThreshold(threshold=_vt)
#         sc = MinMaxScaler()
#         sc.fit(filtered_df.transpose().to_numpy())
#         sc_df_real = sc.transform(filtered_df.transpose().to_numpy())
#         sc_df_rand = sc.transform(filtered_rand.transpose().to_numpy())
#         vt.fit(sc_df_real)
#         vt_df_real = pd.DataFrame(vt.transform(sc_df_real))
#         vt_df_rand = pd.DataFrame(vt.transform(sc_df_rand))
#         nocorr_real = corrX_new(vt_df_real,cut=corr_cut) #Gives mask for columns to drop
#         noc_real = vt_df_real.iloc[:,nocorr_real] #Columns are features still
#         noc_rand = vt_df_rand.iloc[:,nocorr_real]
#         processed_rand_feats = pd.DataFrame(np.transpose(noc_rand.to_numpy()),columns=filtered_df.columns) #Ensures labels stripped; gives transposed arrays (row = feature, column= instance)
#         processed_real_feats = pd.DataFrame(np.transpose(noc_real.to_numpy()),columns=filtered_df.columns)
#         output_rand = tuple([processed_rand_feats[lbl] for lbl in labels])
#         output_real = tuple([processed_real_feats[lbl] for lbl in labels])
#         return output_rand,output_real

# def outsamp_splits(data_df: pd.DataFrame,num_coup=5,save_mask=True,val_int=True,val_split=10,test_list = None):
#     """
#     Split dataset to withhold specific plates.

#     Get split handles in tuple.

#     Validation boolean decides if output is (train,validate,test)

#     The num_coup integer indicates the number of am_br reactant combinations to withhold into the
#     validate or test split (each)

#     Val split ignored unless val_int is True

#     test_list overrides num_coup, and sets those couplings as the out of sample ones
#     note: only works with internal validation

#     """
#     # no_exp = len(data_df.index)
#     # rand_arr = np.random.randint(1,high=fold+1,size=no_exp,dtype=int)
#     # if validation == False:
#     #     train_mask = (rand_arr > 1).tolist()
#     #     test_mask = (rand_arr == 1).tolist()
#     #     mask_list = [train_mask,test_mask]
#     # elif validation == True:
#     #     train_mask = (rand_arr > 2).tolist()
#     #     validate_mask = (rand_arr == 2).tolist()
#     #     test_mask = (rand_arr == 1).tolist()
#     #     mask_list = [train_mask,validate_mask,test_mask]
#     # out = tuple([data_df.iloc[msk,:] for msk in mask_list])
#     # return out
#     if val_int==False:
#         handles = data_df.index
#         reacts = [f.rsplit('_',3)[0] for f in handles]
#         set_ = sorted(list(set(reacts)))
#         if test_list == None:
#             test = random.sample(set_,num_coup)
#         elif type(test_list) == list:
#             test = test_list
#         temp = [f for f in set_ if f not in test] #temp is train and val
#         val = random.sample(temp,num_coup) #val is sampling of temp (train + val)
#         train = [f for f in temp if f not in val] #train is temp if not in val
#         tr_h = [f for f in handles if f.rsplit('_',3)[0] in train]
#         va_h = [f for f in handles if f.rsplit('_',3)[0] in val]
#         te_h = [f for f in handles if f.rsplit('_',3)[0] in test]
#         mask_list = [tr_h,va_h,te_h]
#         if save_mask == False:
#             out = tuple([data_df.loc[msk,:] for msk in mask_list])
#             return out
#         if save_mask == True:
#             out = tuple([data_df.loc[msk,:] for msk in mask_list]+[val,test])
#             return out
#     elif val_int == True: ##This is to keep test plates out of sample, BUT validation and train data from shared plates. This may be necessary on account of the stochasticity of modeling
#         handles = data_df.index
#         reacts = [f.rsplit('_',3)[0] for f in handles]
#         set_ = sorted(list(set(reacts)))
#         if test_list == None: #This is for randomly sampling a number of couplings ONLY IF TEST_LIST NOT SPECIFIED
#             test = random.sample(set_,num_coup)
#         elif type(test_list) == list: #If test_list is specified, then this overrides everything else
#             test = test_list
#         te_h = [f for f in handles if f.rsplit('_',3)[0] in test]
#         temp = [f for f in handles if f.rsplit('_',3)[0] not in test] #both train and val will come from here; handles, not am_br
#         # print(np.rint(len(temp)/val_split))
#         va_h = random.sample(temp,int(np.rint(len(temp)/val_split))) #handles sampled randomly from train&val list of handles (temp)
#         tr_h = [f for f in temp if f not in va_h]
#         print('check :',[f for f in tr_h if f in va_h or f in te_h]) #data leakage test
#         mask_list = [tr_h,va_h,te_h]
#         if save_mask == False:
#             out = tuple([data_df.loc[msk,:] for msk in mask_list])
#             return out
#         if save_mask == True:
#             out = tuple([data_df.loc[msk,:] for msk in mask_list]+[test])
#             return out

# def outsamp_by_handle(data_df: pd.DataFrame,test_list = []):
#     """
#     No validation; just gives train/test using passed handles for test examples.

#     """
#     handles = data_df.index
#     train_list = [f for f in handles if f not in test_list]
#     test = data_df.loc[test_list,:]
#     train = data_df.loc[train_list,:]
#     return train,test

# def split_handles_reactants(reacts=[],handle_position: int=1,handles=[]):
#     """
#     Partition dataset to withhold specific REACTANTS; flexible to any
#     specified position in reaction handle (amine, bromide, catalyst, etc) NOTE: ONE-INDEXED
#     """
#     str_reacts = [str(f) for f in reacts]
#     out_hand = [f for f in handles if f.strip().split('_')[handle_position-1] in str_reacts] #clean up whitespace, split handles, check specific position for match
#     return out_hand

# def split_outsamp_reacts(dataset_:pd.DataFrame,amines=[],bromides=[],separate=False):
#     """
#     Use this to split out of sample reactants in dataset partitions.

#     This runs split_out_reactants

#     Data should be row = instance

#     "separate" boolean triggers optional output with specific handles for out of sample amines OR bromides
#     """
#     amine_out_hand = split_handles_reactants(reacts=amines,handle_position=1,handles=dataset_.index)
#     # print(amine_out_hand)
#     bromide_out_hand = split_handles_reactants(reacts=bromides,handle_position=2,handles=dataset_.index)
#     # print(bromide_out_hand)
#     outsamp_handles = sorted(list(set(amine_out_hand+bromide_out_hand))) #remove duplicates (from any matches to both reactants) and provide consistent ordering
#     if separate == False: return outsamp_handles
#     elif separate == True:
#         am_f = []
#         br_f = []
#         comb = [str(f[0])+'_'+str(f[1]) for f in itertools.product(amines,bromides)]
#         # print(comb)
#         both = [f for f in outsamp_handles if f.strip().rsplit('_',3)[0] in comb]
#         not_both = [f for f in outsamp_handles if f not in both]
#         for k in amines:
#             temp1 = split_handles_reactants(reacts=[str(k)],handle_position=1,handles=not_both)
#             am_f.append(temp1)
#         for m in bromides:
#             temp2 = split_handles_reactants(reacts=[str(m)],handle_position=2,handles=not_both)
#             br_f.append(temp2)
#         return am_f,br_f,both,outsamp_handles


# if __name__ == '__main__':

#     datafile = r"/home/nir2/tfwork/ROCHE_ws/ROCHE_data_update_thru_9_27_22_checked_noval.csv"
#     data_df = pd.read_csv(datafile,index_col=0,header=None)
#     data_df = cleanup_handles(data_df)
#     handles = data_df.index
#     uni_coup = sorted(list(set([f.rsplit('_',3)[0] for f in handles])))
#     ## Amines
#     col1_ = ml.Collection.from_zip("descriptors/confs_am_optgfn2.zip")
#     col2_ = ml.Collection.from_zip("descriptors/rocheval_amine_conf_am_optgfn2.zip")
#     atomprop_am_dict = pickle.load(open("descriptors/amine_pickle_dict.p",'rb'))
#     sub_am_dict = retrieve_amine_rdf_descriptors(col2_,atomprop_am_dict,increment=0.75)
#     new_am = retrieve_amine_rdf_descriptors(col1_,atomprop_am_dict,increment=0.75)
#     sub_am_dict.update(new_am)

#     ## Bromides
#     col1 = ml.Collection.from_zip("descriptors/confs_br_optgfn2.zip")
#     col2 = ml.Collection.from_zip("descriptors/rocheval_amine_conf_br_optgfn2.zip")
#     atomprop_dict = pickle.load(open("descriptors/bromide_pickle_dict.p",'rb'))
#     sub_br_dict = retrieve_bromide_rdf_descriptors(col1,atomprop_dict,increment=0.75)
#     new_dict = retrieve_bromide_rdf_descriptors(col2,atomprop_dict,increment=0.75)
#     sub_br_dict.update(new_dict)

#     ############################################
#     ### Control for sub-only random features ###
#     ############################################
#     directory = 'descriptors/'
#     basefile = directory+'base_params.csv'
#     basedf = pd.read_csv(basefile,header=None,index_col=0).transpose()
#     solvfile = directory+'solvent_params.csv'
#     solvdf = pd.read_csv(solvfile,header=None,index_col=0).transpose()
#     # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ## Use this for ASO/AEIF
#     catfile = r"/home/nir2/tfwork/ROCHE_ws/descriptors/iso_catalyst_embedding.csv" ##Use this for embedded catalysts
#     catdf = pd.read_csv(catfile,header=None,index_col=0).transpose()
#     ### Am,Br,Cat,Solv,Base is order for desc
#     ############################################

#     #######################
#     ### SUB ONLY RANDOM ###
#     #######################

#     rand_all = make_randomized_features(sub_am_dict,sub_br_dict,catfile=catfile)
#     rand = rand_all
#     # rand = (rand_all[0],rand_all[1],catdf,solvdf,basedf)

#     #######################
#     ### NORMAL ############
#     #######################
#     # rand = make_randomized_features(sub_am_dict,sub_br_dict)
#     #######################


#     model_dir_real = r"/home/nir2/tfwork/ROCHE_ws/Nov-19-2022-00-00_realcont10_outval_outte/out/"
#     directory_real = r"/home/nir2/tfwork/ROCHE_ws/Nov-18-2022out_te_samepre_OUTsampval_075inc_vt03_maincont/real"
#     model_dir_rand = r"/home/nir2/tfwork/ROCHE_ws/Nov-19-2022-00-00_randcont10_outval_outte/out/"
#     directory_rand = r"/home/nir2/tfwork/ROCHE_ws/Nov-18-2022out_te_samepre_OUTsampval_075inc_vt03_maincont/rand"

#     organ_real = tf_organizer('real',partition_dir=directory_real)
#     drive_real = tfDriver(organ_real)

#     organ_rand = tf_organizer('rand',partition_dir=directory_rand)
#     drive_rand = tfDriver(organ_rand)

#     date__ = date.today().strftime("%b-%d-%Y")

#     out_dir_path = r'/home/nir2/tfwork/ROCHE_ws/nn_val_controlfinal_'+date__+'_updated/out/'
#     os.makedirs(out_dir_path,exist_ok=True)

#     metrics = []
#     predictions_real = []
#     predictions_rand = []

#     train_val_test = {}
#     train_val_test['train']=[]
#     train_val_test['val']=[]
#     train_val_test['test']=[]
#     val_hand_f = r"/home/nir2/tfwork/ROCHE_ws/validation_handles.csv"

#     ser = pd.read_csv(val_hand_f)
#     # print(len(ser.values))
#     hand = [f[0].strip() for f in ser.values.tolist()]
#     # print(hand)

#     # pred_str = ','.join(hand)

#     x_p_df = pd.read_feather(r"/home/nir2/tfwork/ROCHE_ws/Nov-21-2022valonly_desc_catemb_075inc_vt03/real/valonly_prophetic_xp.feather") #REAL descriptors
#     pred_idx = x_p_df.columns.tolist()
#     x_p_re = assemble_descriptors_from_handles(pred_idx,sub_am_dict,sub_br_dict) #real
#     x_p_ra = assemble_random_descriptors_from_handles(pred_idx,rand) #rand

#     with open('prediction_buffer_real'+date__+'control.csv','a') as g:
#         g.write(','.join(pred_idx)+'\n')

#     with open('prediction_buffer_rand'+date__+'control.csv','a') as g:
#         g.write(','.join(pred_idx)+'\n')

#     # gpus = tf.config.experimental.list_physical_devices('GPU')
#     # try:
#     #     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=46000)])
#     # except RuntimeError as e:
#     #     print(e)
#     with tf.device('/GPU:0'):
#         for __k in range(len(drive_real.organizer.partitions)):
#             ### Feature stuff ###
#             xtr,xval,xte,ytr,yval,yte = [f[0] for f in drive_real.x_y]
#             name_ = str(drive_real.current_part_id)
#             print(name_)
#             partition_index = organ_real.partIDs.index(int(name_))
#             tr,va,te,y1,y2,y3 = drive_real.organizer.partitions[partition_index]
#             tr,va,te = drive_real._feather_to_np((tr,va,te))
#             # print(tr[0].shape,x_p_df.shape)
#             x_tr = assemble_descriptors_from_handles(tr[1].to_list(),sub_am_dict,sub_br_dict)
#             x_va = assemble_descriptors_from_handles(va[1].to_list(),sub_am_dict,sub_br_dict)
#             x_te = assemble_descriptors_from_handles(te[1].to_list(),sub_am_dict,sub_br_dict)
#             # (x_tr_,x_va_,x_te_,x_p_) = preprocess_feature_arrays((x_tr,x_va,x_te,x_p),save_mask=False)
#             x_tr_ra = assemble_random_descriptors_from_handles(tr[1].to_list(),rand)
#             x_va_ra = assemble_random_descriptors_from_handles(va[1].to_list(),rand)
#             x_te_ra = assemble_random_descriptors_from_handles(te[1].to_list(),rand)
#             (x_tr_,x_va_,x_te_,x_p_),(x_tr_ra_,x_va_ra_,x_te_ra_,x_p_ra_) = new_mask_random_feature_arrays((x_tr,x_va,x_te,x_p_re),(x_tr_ra,x_va_ra,x_te_ra,x_p_ra),_vt=1e-3)

#             ### Real ###
#             train_val_test['train']=[]
#             train_val_test['val']=[]
#             train_val_test['test']=[]
#             models__ = glob(model_dir_real+name_+"hpset*.h5")
#             # print(models__)
#             ytr = ytr.ravel()
#             yval = yval.ravel()
#             yte= yte.ravel()
#             train_val_test['train'].append(ytr)
#             train_val_test['val'].append(yval)
#             train_val_test['test'].append(yte)
#             for _model in models__:
#                 ### For new inferences
#                 model_config = keras.models.load_model(_model).get_config()
#                 # print(model_config)
#                 model_config['layers'][0]['config']['batch_input_shape'] = (None,x_tr_.transpose().shape[1]) #Update shape to match NEW preprocessing
#                 model = tf.keras.Sequential.from_config(model_config)
#                 model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1),
#                         loss = 'mse',
#                         metrics=['accuracy','mean_absolute_error','mean_squared_error'],
#                 )
#                 history = model.fit(
#                     x_tr_.transpose().to_numpy(),
#                     ytr,
#                     batch_size=32,
#                     epochs=175,
#                     validation_data=(x_va_.transpose().to_numpy(),yval),
#                     verbose=False,
#                     # workers=64,
#                     callbacks = [ModelCheckpoint(filepath=out_dir_path+name_+'real_best_model.h5',
#                                 monitor='val_loss',
#                                 save_best_only=True
#                                 )]
#                 )
#                 model.load_weights(out_dir_path+name_+'real_best_model.h5')
#                 ### For getting model values to plot
#                 # model = keras.models.load_model(_model)
#                 xtr = x_tr_.transpose().to_numpy()
#                 xval = x_va_.transpose().to_numpy()
#                 xte = x_te_.transpose().to_numpy()
#                 x_p = x_p_.transpose().to_numpy() #Important to get index-instance column-features
#                 ytr_p,yval_p,yte_p,yp_p = model_inference(model,xtr,(xval,xte,x_p))
#                 print(yp_p.shape,'real')
#                 predictions_real.append(yp_p)
#                 train_val_test['train'].append(ytr_p)
#                 train_val_test['val'].append(yval_p)
#                 train_val_test['test'].append(yte_p)
#                 mae_tr = mean_absolute_error(ytr,ytr_p)
#                 mae_te = mean_absolute_error(yte,yte_p)
#                 mae_val = mean_absolute_error(yval,yval_p)
#                 metrics.append([mae_tr,mae_val,mae_te])
#                 with open('prediction_buffer_real'+date__+'control.csv','a') as g:
#                     g.write(','.join([str(f) for f in yp_p])+'\n')
#             outdf = pd.DataFrame(train_val_test['train']).transpose()
#             outdf.index = x_tr_.columns
#             valdf = pd.DataFrame(train_val_test['val']).transpose()
#             valdf.index = x_va_.columns
#             tedf = pd.DataFrame(train_val_test['test']).transpose()
#             tedf.index = x_te_.columns
#             outdf_ = pd.concat((outdf,valdf,tedf),axis=0,keys=['train','val','test'])
#             outdf_.to_csv(out_dir_path+'output_real_models_part'+name_+'_'+'_'.join([str(len(ytr)),
#                             str(len(yval)),str(len(yte))])+'.csv')

#             ### Rand ###
#             train_val_test['train']=[] #clear cache
#             train_val_test['val']=[]
#             train_val_test['test']=[]
#             models__ = glob(model_dir_rand+name_+"hpset*.h5") #get random feature models
#             print(models__)
#             ytr = ytr.ravel() #Y is same between random and real - these are made at the same time
#             yval = yval.ravel()
#             yte= yte.ravel()
#             train_val_test['train'].append(ytr)
#             train_val_test['val'].append(yval)
#             train_val_test['test'].append(yte)
#             for _model in models__:
#                 ### For new inferences
#                 model_config = keras.models.load_model(_model).get_config()
#                 # print(model_config)
#                 model_config['layers'][0]['config']['batch_input_shape'] = (None,x_tr_.transpose().shape[1]) #random shape SHOULD be the same; if it is not, something is wrong
#                 model = tf.keras.Sequential.from_config(model_config)
#                 model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1),
#                         loss = 'mse',
#                         metrics=['accuracy','mean_absolute_error','mean_squared_error'],
#                 )
#                 history = model.fit(
#                     x_tr_ra_.transpose().to_numpy(), #This is fit on the random features
#                     ytr,
#                     batch_size=32,
#                     epochs=175,
#                     validation_data=(x_va_ra_.transpose().to_numpy(),yval), #This is validated on random features
#                     verbose=False,
#                     callbacks = [ModelCheckpoint(filepath=out_dir_path+name_+'rand_best_model.h5',
#                                 monitor='val_loss',
#                                 save_best_only=True
#                                 )]
#                 )
#                 model.load_weights(out_dir_path+name_+'rand_best_model.h5')
#                 ### For getting model values to plot
#                 # model = keras.models.load_model(_model)
#                 ## Prep random features for inferences to get final values ##
#                 xtr = x_tr_ra_.transpose().to_numpy()
#                 xval = x_va_ra_.transpose().to_numpy()
#                 xte = x_te_ra_.transpose().to_numpy()
#                 x_p = x_p_ra_.transpose().to_numpy() #Important to get index-instance column-features
#                 ytr_p,yval_p,yte_p,yp_p = model_inference(model,xtr,(xval,xte,x_p))
#                 print(yp_p.shape,'rand')
#                 predictions_rand.append(yp_p)
#                 train_val_test['train'].append(ytr_p)
#                 train_val_test['val'].append(yval_p)
#                 train_val_test['test'].append(yte_p)
#                 mae_tr = mean_absolute_error(ytr,ytr_p)
#                 mae_te = mean_absolute_error(yte,yte_p)
#                 mae_val = mean_absolute_error(yval,yval_p)
#                 metrics.append([mae_tr,mae_val,mae_te])
#                 with open('prediction_buffer_rand'+date__+'control.csv','a') as g:
#                     g.write(','.join([str(f) for f in yp_p])+'\n')
#             outdf = pd.DataFrame(train_val_test['train']).transpose()
#             outdf.index = x_tr_ra_.columns
#             valdf = pd.DataFrame(train_val_test['val']).transpose()
#             valdf.index = x_va_ra_.columns
#             tedf = pd.DataFrame(train_val_test['test']).transpose()
#             tedf.index = x_te_ra_.columns
#             outdf_ = pd.concat((outdf,valdf,tedf),axis=0,keys=['train','val','test'])
#             outdf_.to_csv(out_dir_path+'output_rand_models_part'+name_+'_'+'_'.join([str(len(ytr)),
#                             str(len(yval)),str(len(yte))])+'.csv')
#             drive_real.get_next_part()
#             print('completed part number: '+str(__k))


#     pred_df_real = pd.DataFrame(predictions_real,columns=pred_idx).transpose()
#     pred_df_real.to_csv(out_dir_path+'prophetic_output_real'+date__+'.csv')
#     pred_df_rand = pd.DataFrame(predictions_rand,columns=pred_idx).transpose()
#     pred_df_rand.to_csv(out_dir_path+'prophetic_output_rand'+date__+'.csv')
