####### Reference scripts for inferencing


# import os
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# from glob import glob
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# from scipy.stats import linregress
# from sklearn.metrics import mean_absolute_error
# from prototype_roche_nn_screen import tf_organizer, tfDriver
# from datetime import date
# import pickle
# import json

# # import new_rdf_manual as desc
# import molli as ml
# import sys
# from keras.callbacks import EarlyStopping, ModelCheckpoint


# def get_mae_metrics(
#     model, X: np.array, inference_x: (np.array), y: np.array, infer_labels: (np.array)
# ):
#     """
#     Get inferences on partitions, then evaluate errors and report mae

#     2/26/2022 VAL ONLY

#     """
#     yva, yte = infer_labels
#     train_errors, val_errors, test_errors = compute_residuals(
#         model, X, inference_x, y, (yva, yte)
#     )
#     trainmae = np.mean(train_errors)
#     valmae = np.mean(val_errors)
#     testmae = np.mean(test_errors)
#     return trainmae, valmae, testmae


# def get_mse_metrics(
#     model, X: np.array, inference_x: (np.array), y: np.array, infer_labels: (np.array)
# ):
#     """
#     Get inferences, then evaluate mse

#     2/26/2022 VAL ONLY

#     """
#     yva, yte = infer_labels
#     train_errors, val_errors, test_errors = compute_residuals(
#         model, X, inference_x, y, (yva, yte)
#     )
#     trainmse = np.sqrt(np.sum(np.square(train_errors))) / len(train_errors)
#     valmse = np.sqrt(np.sum(np.square(val_errors))) / len(val_errors)
#     testmse = np.sqrt(np.sum(np.square(test_errors))) / len(test_errors)
#     return trainmse, valmse, testmse


# def compute_residuals(model, X, inference_x: (np.array), y, infer_labels: (np.array)):
#     """
#     Get residual errors for partitions

#     2/26/2022 VAL ONLY

#     """
#     yva, yte = infer_labels
#     ytr_p, yva_p, yte_p = model_inference(model, X, inference_x)
#     train_errors = abs(ytr_p - y)
#     val_errors = abs(yva_p - yva)
#     test_errors = abs(yte_p - yte)
#     return train_errors, val_errors, test_errors


# def model_inference(model, X, inference_x: (np.array)):
#     """
#     Takes inference tuple, and processes it. This has val , test, and prophetic X.

#     Use trained model (or instantiated from identified parameters)

#     Outputs predicted values based on descriptors
#     """
#     if len(inference_x) == 3:
#         X_val, X_test, Xp = inference_x
#         ytr_p = model.predict(X).ravel()
#         yte_p = model.predict(X_test).ravel()
#         yva_p = model.predict(X_val).ravel()
#         ypr_p = model.predict(Xp).ravel()
#         return ytr_p, yva_p, yte_p, ypr_p
#     elif len(inference_x) == 2:
#         X_val, X_test = inference_x
#         ytr_p = model.predict(X).ravel()
#         yte_p = model.predict(X_test).ravel()
#         yva_p = model.predict(X_val).ravel()
#         return ytr_p, yva_p, yte_p
#     elif len(inference_x) == 1:
#         X_test = inference_x
#         ytr_p = model.predict(X).ravel()
#         yte_p = model.predict(X_test).ravel()
#         return ytr_p, yte_p
#     else:
#         raise Exception("Pass inference array; did not get proper number")


# def _trans_xy_(desc: pd.DataFrame):
#     """
#     Output feature arrays from input dataframes.

#     Note: this is designed for an ecosystem which puts instances as columns and features as
#     rows.

#     Returns array and labels
#     """
#     transposition = desc.transpose()
#     if type(transposition.index[0]) != str:
#         raise ValueError("Check dataframe construction")
#     feature_array = transposition.to_numpy()
#     return feature_array, transposition.index


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


# def plot_results(outdir: str, expkey: str, train: (np.array), test: (np.array)):
#     """
#     Plot model predicted vs observed as an image.
#     """
#     observed, predicted = test
#     # print(observed,predicted)
#     traino, trainp = train
#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1)
#     plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     plt.plot(traino, trainp, "ko")
#     plt.title("Train")
#     plt.ylabel("Predicted")
#     plt.xlabel("Observed")
#     plt.text(90, 3, str(mean_absolute_error(traino, trainp)))
#     k, b, r, p, _ = linregress(traino, trainp)
#     plt.plot([0, 100], [0 * k + b, 100 * k + b], alpha=0.65)
#     plt.text(
#         75.0,
#         7.0,
#         f"$ \\langle R^2 \\rangle = {r**2:0.4f} $ \n $ \\langle k \\rangle = {k:0.3f} $",
#     )
#     fig.add_subplot(1, 2, 2)
#     plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     plt.plot(observed, predicted, "rp")
#     plt.title("Test")
#     plt.xlabel("Observed")
#     plt.ylabel("Predicted")
#     plt.text(90, 3, str(mean_absolute_error(observed, predicted)))
#     k, b, r, p, _ = linregress(observed, predicted)
#     plt.plot([0, 100], [0 * k + b, 100 * k + b], alpha=0.60)
#     plt.text(
#         75.0,
#         7.0,
#         f"$ \\langle R^2 \\rangle = {r**2:0.4f} $ \n $ \\langle k \\rangle = {k:0.3f} $",
#     )
#     plt.savefig(outdir + expkey + ".png")


# import os
# # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import tensorflow as tf
# # tf.config.set_visible_devices([],'GPU')
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# from glob import glob
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import VarianceThreshold,SequentialFeatureSelector
# from sklearn.linear_model import RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# from scipy.stats import linregress
# from sklearn.metrics import mean_absolute_error
# from prototype_roche_nn_screen import tf_organizer,tfDriver
# from datetime import date
# import pickle
# import json
# # import new_rdf_manual as desc
# import randfeat_rdf_manual as desc
# import molli as ml
# import sys
# from keras.callbacks import EarlyStopping,ModelCheckpoint


# # model = keras.models.load_model("/home/nir2/tfwork/ROCHE_ws/Feb-27-2022-00-00/out/1005hpset0_5504n_584n_208n_0.3428575535893392d_0.3224495289586471d_gelua_softmaxa_softmaxa.h5")


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


# # y_train = sorted(glob("/home/nir2/tfwork/ROCHE_ws/partitions/"+'/1005_*ytr.feather'))[0]
# # y_te = sorted(glob("/home/nir2/tfwork/ROCHE_ws/partitions/"+'/1005_*yte.feather'))[0]
# # x_train = sorted(glob("/home/nir2/tfwork/ROCHE_ws/partitions/"+'/1005_*xtr.feather'))[0]
# # x_te = sorted(glob("/home/nir2/tfwork/ROCHE_ws/partitions/"+'/1005_*xte.feather'))[0]
# # y_val = sorted(glob("/home/nir2/tfwork/ROCHE_ws/partitions/"+'/1005_*yva.feather'))[0]
# # x_val = sorted(glob("/home/nir2/tfwork/ROCHE_ws/partitions/"+'/1005_*xva.feather'))[0]

# # # dfs = tfDriver._feather_to_np((x_train,x_val,x_te,y_train,y_val,y_te))

# # # print(dfs)

# # # list_of_arrays = []
# # # list_of_labels = []
# # # for i,j in dfs:
# # #     list_of_arrays.append(i)
# # #     list_of_labels.append(j)


# # dfs = _feather_to_np((x_train,x_val,x_te,y_train,y_val,y_te))

# # # print(dfs)

# # list_of_arrays = []
# # list_of_labels = []
# # label_dict = {}
# # for i,j in dfs:
# #     list_of_arrays.append(i)
# #     list_of_labels.append(j)


# # # models = glob(r"G:\hd5_models_antares/*.h5")

# # # model = keras.models.load_model(r"C:\Users\irine\denmark\ROCHE\modeling\prototyping\1005hpset0_5504n_584n_208n_0.3428575535893392d_0.3224495289586471d_gelua_softmaxa_softmaxa.h5")
# # # list_of_arrays_ = [f.ravel() for f in list_of_arrays]
# # xtr,xva,xte,ytr,yva,yte = list_of_arrays

# # ytr = ytr.ravel()
# # yva = yva.ravel()
# # yte= yte.ravel()

# # print(xtr.shape)

# # filter = VarianceThreshold(threshold=0.10).fit(xtr)
# # xtr = filter.transform(xtr)
# # xva = filter.transform(xva)
# # xte = filter.transform(xte)

# # print(len(xtr[0]))


# ### Feature selection

# # n_feat = 350
# # estimator = RidgeCV()
# # selector = SequentialFeatureSelector(estimator,n_features_to_select=n_feat,direction='backward',cv=5,n_jobs=8)
# # selector.fit(xtr,ytr)
# # xtr = selector.transform(xtr)
# # xva = selector.transform(xva)
# # xte = selector.transform(xte)

# ### Plotting sklearn model

# # expkey_ = '1005_test_variance_thresh'

# # xtr_sel = pd.DataFrame(xtr,index=list_of_labels[0]).transpose()
# # xva_sel = pd.DataFrame(xva,index=list_of_labels[1]).transpose()
# # xte_sel = pd.DataFrame(xte,index=list_of_labels[2]).transpose()
# # xtr_sel.to_feather('SFS_'+expkey_+'xtr.feather')
# # xva_sel.to_feather('SFS_'+expkey_+'xva.feather')
# # xte_sel.to_feather('SFS_'+expkey_+'xte.feather')


# # model = RandomForestRegressor(n_estimators=750,n_jobs=4,random_state=111,max_features="auto",max_depth=8,max_leaf_nodes=8,min_samples_split=2)
# # # model = RidgeCV(cv=3)
# # model.fit(xtr,ytr)

# # ytr_p,yval_p,yte_p = model_inference(model,xtr,(xva,xte))

# # print(ytr_p.shape,yte_p.shape,list_of_arrays[3].shape,list_of_arrays[5].shape)
# # plot_results(outdir='./',expkey='test_part1005model',train=(ytr,ytr_p),test=(yte,yte_p))


# # model_dir = r"/home/nir2/tfwork/ROCHE_ws/Mar-14-2022-00-00/out/"
# # directory_ = r"/home/nir2/tfwork/ROCHE_ws/partitions_scaled_inc085"

# model_dir = r"/home/nir2/tfwork/ROCHE_ws/Nov-19-2022-00-00_realcont10_outval_outte/out/"
# directory_ = r"/home/nir2/tfwork/ROCHE_ws/Nov-18-2022out_te_samepre_OUTsampval_075inc_vt03_maincont/real"

# organ = tf_organizer('blah',partition_dir=directory_)
# drive = tfDriver(organ)


# date__ = date.today().strftime("%b-%d-%Y")+'oldmodels'

# out_dir_path = r'/home/nir2/tfwork/ROCHE_ws/nn_last_three_infer_'+date__+'_forSI/out/'
# os.makedirs(out_dir_path,exist_ok=True)


# metrics = []
# predictions = []

# train_val_test = {}
# train_val_test['train']=[]
# train_val_test['val']=[]
# train_val_test['test']=[]
# col1_ = ml.Collection.from_zip("descriptors/confs_am_optgfn2.zip")
# col2_ = ml.Collection.from_zip("descriptors/rocheval_amine_conf_am_optgfn2.zip")
# atomprop_am_dict = pickle.load(open("descriptors/amine_pickle_dict.p",'rb'))
# sub_am_dict = desc.retrieve_amine_rdf_descriptors(col1_,atomprop_am_dict,increment=0.75)
# new_am = desc.retrieve_amine_rdf_descriptors(col2_,atomprop_am_dict,increment=0.75)
# sub_am_dict.update(new_am)
# col1 = ml.Collection.from_zip("descriptors/confs_br_optgfn2.zip")
# col2 = ml.Collection.from_zip("descriptors/rocheval_amine_conf_br_optgfn2.zip")
# atomprop_dict = pickle.load(open("descriptors/bromide_pickle_dict.p",'rb'))
# sub_br_dict = desc.retrieve_bromide_rdf_descriptors(col1,atomprop_dict,increment=0.75)
# new_dict = desc.retrieve_bromide_rdf_descriptors(col2,atomprop_dict,increment=0.75)
# sub_br_dict.update(new_dict)


# pred_str = sys.argv[1]
# x_p_df = desc.assemble_descriptors_from_handles(pred_str,sub_am_dict,sub_br_dict)
# pred_idx = x_p_df.columns.tolist()

# with open('prediction_buffer_'+date__+'.csv','a') as g:
#     g.write(','.join(pred_idx)+'\n')

# with tf.device('/GPU:0'):
#     for __k in range(len(drive.organizer.partitions)):
#         train_val_test['train']=[]
#         train_val_test['val']=[]
#         train_val_test['test']=[]
#         xtr,xval,xte,ytr,yval,yte = [f[0] for f in drive.x_y]
#         name_ = str(drive.current_part_id)
#         print(name_)
#         partition_index = organ.partIDs.index(int(name_))
#         tr,va,te,y1,y2,y3 = drive.organizer.partitions[partition_index]
#         tr,va,te = drive._feather_to_np((tr,va,te))
#         # print(tr[1])
#         x_tr = desc.assemble_descriptors_from_handles(tr[1].to_list(),sub_am_dict,sub_br_dict)
#         x_va = desc.assemble_descriptors_from_handles(va[1].to_list(),sub_am_dict,sub_br_dict)
#         x_te = desc.assemble_descriptors_from_handles(te[1].to_list(),sub_am_dict,sub_br_dict)
#         (x_tr_,x_va_,x_te_,x_p_) = desc.preprocess_feature_arrays((x_tr,x_va,x_te,x_p_df),save_mask=False)
#         # print(x_tr_.transpose(),tr[0])
#         # print(x_tr_.transpose().shape,tr[0].shape)
#         # print(x_p_)
#         # print(x_tr_,x_tr_.shape)
#         # if x_tr_.shape[0] != tr[0].shape[0]:
#         #     print(x_tr_,tr)
#         #     raise Exception('feature preprocessing failed')
#         models__ = glob(model_dir+name_+"hpset*.h5")
#         print(models__)
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
#                         loss = 'mse',
#                         metrics=['accuracy','mean_absolute_error','mean_squared_error'],
#             )
#             history = model.fit(
#                 x_tr_.transpose().to_numpy(),
#                 ytr,
#                 batch_size=32,
#                 epochs=175,
#                 validation_data=(x_va_.transpose().to_numpy(),yval),
#                 workers=64,
#                 callbacks = [ModelCheckpoint(filepath=out_dir_path+name_+'_best_model.h5',
#                             monitor='val_loss',
#                             save_best_only=True
#                             )]
#             )
#             model.load_weights(out_dir_path+name_+'_best_model.h5')
#             ### For getting model values to plot
#             # model = keras.models.load_model(_model)
#             xtr = x_tr_.transpose().to_numpy()
#             xval = x_va_.transpose().to_numpy()
#             xte = x_te_.transpose().to_numpy()
#             x_p = x_p_.transpose().to_numpy() #Important to get index-instance column-features
#             ytr_p,yval_p,yte_p,yp_p = model_inference(model,xtr,(xval,xte,x_p))
#             predictions.append(yp_p)
#             train_val_test['train'].append(ytr_p)
#             train_val_test['val'].append(yval_p)
#             train_val_test['test'].append(yte_p)
#             mae_tr = mean_absolute_error(ytr,ytr_p)
#             mae_te = mean_absolute_error(yte,yte_p)
#             mae_val = mean_absolute_error(yval,yval_p)
#             metrics.append([mae_tr,mae_val,mae_te])
#             with open('prediction_buffer_'+date__+'.csv','a') as g:
#                 g.write(','.join([str(f) for f in yp_p])+'\n')
#         outdf = pd.DataFrame(train_val_test['train']).transpose()
#         outdf.index = x_tr_.columns
#         valdf = pd.DataFrame(train_val_test['val']).transpose()
#         valdf.index = x_va_.columns
#         tedf = pd.DataFrame(train_val_test['test']).transpose()
#         tedf.index = x_te_.columns
#         outdf_ = pd.concat((outdf,valdf,tedf),axis=0,keys=['train','val','test'])
#         outdf_.to_csv(out_dir_path+'output_models_part'+name_+'_'+'_'.join([str(len(ytr)),
#                         str(len(yval)),str(len(yte))])+'.csv')
#         drive.get_next_part()

# pred_df = pd.DataFrame(predictions,columns=pred_idx).transpose()
# pred_df.to_csv(out_dir_path+'prophetic_output_'+date__+'.csv')
