from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import json
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xarray as xr
import glob
from utils import db2lin, lin2db


def read_input_data(input_path, s1_variables, s2_variables, crop_list, pattern):
    s2_variable_dic = {k: [] for k in s2_variables}
    s1_variable_dic = {k: [] for k in s1_variables}
    field_ids_list = []
    crop_types_list = []
    #create to match crop types to numeric values
    crop_dic = {crop: crop_list.index(crop) for crop in crop_list}
    counter = 0
    for filename in glob.glob(os.path.join(input_path, '*{}*.nc'.format(pattern))):
        # print(filename)
        xds = xr.open_dataset(filename, engine="netcdf4")
        if xds.crop_type in crop_list:
            counter += 1
            #print(counter)
            for s2_variable in s2_variables:
                timesteps_points = xds['time']
                s2_ts = xds[s2_variable].load().resample(time='6D').mean()
                if s2_ts.shape[0] == 61:
                    s2_variable_dic[s2_variable].append(np.expand_dims(s2_ts, -1))
            for s1_variable in s1_variables:
                timesteps_points = xds['time']
                s1_ts = xr.apply_ufunc(lin2db, xr.apply_ufunc(db2lin, xds[s1_variable].load()).resample(time='6D').mean())
                #print(s1_ts.shape)
                if s1_ts.shape[0] == 61:
                    s1_variable_dic[s1_variable].append(s1_ts)
                    field_ids_list.append(xds.field_id)
            if s1_ts.shape[0] == 61:
                crop_types_list.append(crop_dic[xds.crop_type])
    all_features = s1_variables+s2_variables
    all_variables_dic = {**s1_variable_dic, **s2_variable_dic}
    feature_array_list = []
    for feature in all_features:
        feature_array_list.append(np.stack(all_variables_dic[feature], axis=0))
    featue_array = np.stack(feature_array_list, axis=2)

    return np.squeeze(featue_array), np.array(crop_types_list), field_ids_list

def crop_classification_workflow(input_path, s1_variables, s2_variables, crop_list, pattern, batch_size, epochs,
                                 end_indx):
    featue_array, crop_type_array, field_ids_list = read_input_data(input_path, s1_variables, s2_variables,
                                                                    crop_list, pattern)

    values, counts = np.unique(crop_type_array, return_counts=True)
    print(values, counts)
    #Shuffle the dataset
    idxes = np.random.permutation(len(featue_array))
    featue_array, crop_type_array = featue_array[idxes], crop_type_array[idxes]
    field_ids_list = [field_ids_list[i] for i in idxes]

    #Hot encode labels
    crop_type_array = to_categorical(crop_type_array, num_classes=len(values))

    featue_array = np.nan_to_num(featue_array)

    featue_array = featue_array[:, :end_indx, :]

    #Normalize data
    scalers = {}
    for i in range(featue_array.shape[2]):
        scalers[i] = preprocessing.MinMaxScaler()
        featue_array[:, :, i] = scalers[i].fit_transform(featue_array[:, :, i])

    # split dataset into 0.6 training, 0.2 testing, 0.2 validation
    # X_train, X_test, Y_train, Y_test, field_ids_train, field_ids_test = train_test_split(featue_array, crop_type_array,
    #                                                                                  field_ids_list, test_size=0.2,
    #                                                                                  random_state=42)
    X_train, X_val, Y_train, Y_val, field_ids_train, field_ids_val = train_test_split(featue_array, crop_type_array, field_ids_list,
                                                                                  test_size=0.25, random_state=42)

    #set-up model
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=400, input_shape=(X_train.shape[1], X_train.shape[2]),
                                                          kernel_regularizer=l2(0.01), return_sequences=True,
                                                          recurrent_dropout=0.6)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=200, kernel_regularizer=l2(0.01),
                                                                return_sequences=False,
                                                                 recurrent_dropout=0.4)))
    model.add(tf.keras.layers.Dense(len(values), activation='softmax'))

    opt = tf.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    steps_per_epoch = X_train.shape[0] / batch_size
    history = model.fit(X_train, Y_train, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, callbacks=[callback],
                        validation_data=(X_val, Y_val))
    model.save(r"D:\YIPEEO\analysis\wp4_cropclassification\models\model_sig0_{}.hd5".format(str(end_indx)))

def make_prediction(input_path, s1_variables, s2_variables, crop_list, pattern, end_indx, path2model):
    featue_array, crop_type_array, field_ids_list = read_input_data(input_path, s1_variables, s2_variables,
                                                                    crop_list, pattern)

    values, counts = np.unique(crop_type_array, return_counts=True)
    print(values, counts)

    featue_array = np.nan_to_num(featue_array)

    featue_array = featue_array[:, :end_indx, :]

    # Normalize data
    scalers = {}
    for i in range(featue_array.shape[2]):
        scalers[i] = preprocessing.MinMaxScaler()
        featue_array[:, :, i] = scalers[i].fit_transform(featue_array[:, :, i])

    model = load_model(path2model)
    y_pred = model.predict(featue_array)
    y_pred = np.argmax(y_pred, axis=1)
    cl_report = classification_report(crop_type_array, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(crop_type_array, y_pred, labels=None, sample_weight=None, normalize=None)
    np.savetxt(r"D:\YIPEEO\analysis\wp4_cropclassification\cl_reports\conf_matrix_cz_{}_2018.txt".format(str(end_indx)), conf_matrix, delimiter=",")
    outfile = open(r"D:\YIPEEO\analysis\wp4_cropclassification\cl_reports\cl_report_cz_{}_2018.txt".format(str(
        end_indx)),
                   "w")

    # write file
    outfile.write(json.dumps(cl_report))

    # close file
    outfile.close()


def transfer_learning(input_path, s1_variables, s2_variables, crop_list, pattern, end_indx, path2model):
    featue_array, crop_type_array, field_ids_list = read_input_data(input_path, s1_variables, s2_variables,
                                                                    crop_list, pattern)

    values, counts = np.unique(crop_type_array, return_counts=True)
    print(values, counts)

    #Shuffle the dataset
    idxes = np.random.permutation(len(featue_array))
    featue_array, crop_type_array = featue_array[idxes], crop_type_array[idxes]
    field_ids_list = [field_ids_list[i] for i in idxes]

    featue_array = np.nan_to_num(featue_array)

    featue_array = featue_array[:, :end_indx, :]

    # Normalize data
    scalers = {}
    for i in range(featue_array.shape[2]):
        scalers[i] = preprocessing.MinMaxScaler()
        featue_array[:, :, i] = scalers[i].fit_transform(featue_array[:, :, i])

    model = load_model(path2model)

    # Normalize data
    scalers = {}
    for i in range(featue_array.shape[2]):
        scalers[i] = preprocessing.MinMaxScaler()
        featue_array[:, :, i] = scalers[i].fit_transform(featue_array[:, :, i])

    # split dataset into 0.6 training, 0.2 testing, 0.2 validation
    # X_train, X_test, Y_train, Y_test, field_ids_train, field_ids_test = train_test_split(featue_array, crop_type_array,
    #                                                                                  field_ids_list, test_size=0.2,
    #                                                                                  random_state=42)
    X_train, X_val, Y_train, Y_val, field_ids_train, field_ids_val = train_test_split(featue_array, crop_type_array,
                                                                                      field_ids_list,
                                                                                      test_size=0.5, random_state=42)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    steps_per_epoch = X_train.shape[0] / 16
    history = model.fit(X_train, Y_train, batch_size=16, steps_per_epoch=steps_per_epoch,
                        epochs=5, callbacks=[callback],
                        validation_data=(X_val, Y_val))


    y_pred = model.predict(featue_array)
    y_pred = np.argmax(y_pred, axis=1)
    cl_report = classification_report(crop_type_array, y_pred, output_dict=True)

    outfile = open(r"D:\YIPEEO\analysis\wp4_cropclassification\cl_reports\cl_report_ua_transfer_{}_2018.txt".format(str(
        end_indx)),
                   "w")

    # write file
    outfile.write(json.dumps(cl_report))

    # close file
    outfile.close()


#TODO: define season/temporal extent
#TODo: implement early season classification
if __name__ == '__main__':
    # crop_classification_workflow(r"D:\YIPEEO\data\predictors\s1s2_nc", ["sig40_vv_mean_daily", "sig40_vh_mean_daily",
    #                                                                     "sig40_cr_mean_daily"
    #                                                                     ], ["evi", "ndvi"],
    #                              ["common winter wheat", "grain maize and "
    #                                                                                      "corn-cob-mix",
    #                               "spring barley", "winter barley", "soya", "winter rape and turnip rape seeds"],
    #                              "cz", 16, 150, 20)
    make_prediction(r"D:\YIPEEO\data\predictors\2022_excluded", ["sig40_vv_mean_daily", "sig40_vh_mean_daily",
                                                                        "sig40_cr_mean_daily"
                                                                        ], ["evi", "ndvi"], ["common winter wheat", "grain maize and "
                                                                                         "corn-cob-mix",
                                  "spring barley", "winter barley", "soya", "winter rape and turnip rape seeds"],
                    "cz", 20,
                    r"D:\YIPEEO\analysis\wp4_cropclassification\models\model_sig0_{}.hd5".format(str(20)))

    # transfer_learning(r"D:\YIPEEO\data\predictors\s1s2_nc\ua", ["sig40_vv_mean_daily", "sig40_vh_mean_daily",
    #                                                                     "sig40_cr_mean_daily"
    #                                                                     ], ["evi", "ndvi"], ["common winter wheat", "grain maize and "
    #                                                                                      "corn-cob-mix",
    #                               "spring barley", "winter barley", "soya", "winter rape and turnip rape seeds"],
    #                 "ua", 20,
    #                 r"D:\YIPEEO\analysis\wp4_cropclassification\models\model_sig0_{}.hd5".format(str(20)))