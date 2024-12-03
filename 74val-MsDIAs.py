import numpy as np
from keras.models import load_model
import tensorflow as tf
import time
import os
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def normalize(values, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in values]

def modify_mae(y_true, y_pred):
    floors = y_true[:, -1]
    DI_true = y_true[:, -2]
    mask = tf.sequence_mask(floors, maxlen=y_true.shape[1] - 2)
    y_true_sub1 = tf.ragged.boolean_mask(y_true[:, :-2], mask)
    y_pred_sub1 = tf.ragged.boolean_mask(y_pred[:, :-2], mask)
    StoreyDI_true = tf.exp(y_true_sub1)
    StoreyDI_pred = tf.exp(y_pred_sub1)
    DamageSum_O_true = tf.reduce_sum(StoreyDI_true, axis=1, keepdims=True)
    Lambda_O = StoreyDI_true / DamageSum_O_true
    SDI_PA_O1 = Lambda_O * StoreyDI_pred
    SDI_PA_O = tf.reduce_sum(SDI_PA_O1, axis=1)
    y_pred_sub2 = tf.math.log(SDI_PA_O)
    err1 = tf.reduce_mean(tf.abs(y_true_sub1 - y_pred_sub1), axis=1)
    err2 = tf.abs(DI_true - y_pred_sub2)
    mae = err1 + err2
    mmae_value = tf.reduce_mean(mae)
    return mmae_value

def modify_rmse(y_true, y_pred):
    floors = y_true[:, -1]
    DI_true = y_true[:, -2]
    DI_pred = y_pred[:, -2]
    mask = tf.sequence_mask(floors, maxlen=y_true.shape[1] - 2)
    y_true_sub1 = tf.ragged.boolean_mask(y_true[:, :-2], mask)
    y_pred_sub1 = tf.ragged.boolean_mask(y_pred[:, :-2], mask)
    StoreyDI_true = tf.exp(y_true_sub1)
    StoreyDI_pred = tf.exp(y_pred_sub1)
    DamageSum_O_true = tf.reduce_sum(StoreyDI_true, axis=1, keepdims=True)
    Lambda_O = StoreyDI_true / DamageSum_O_true
    SDI_PA_O1 = Lambda_O * StoreyDI_pred
    SDI_PA_O = tf.reduce_sum(SDI_PA_O1, axis=1)
    y_pred_sub2 = tf.math.log(SDI_PA_O)
    err1 = tf.reduce_mean(tf.square(y_true_sub1 - y_pred_sub1), axis=1)
    err2 = tf.square(DI_true - y_pred_sub2)
    err3 = tf.square(DI_true - DI_pred)
    rmse = tf.sqrt(err1 + err2 + err3)
    mrmse_value = tf.reduce_mean(rmse)
    return mrmse_value

class Dataset:
    def __init__(self, filepath):
        self.train_path = filepath

    def load(self):
        r_data = pd.read_excel(self.train_path)
        r_data_li = r_data.values.tolist()
        input_data = []
        for s_li in r_data_li:
            input_data.append(s_li[0:8])
        print(input_data[0])
        output_data = []
        for s_li in r_data_li:
            output_data.append(s_li[8:21])
        print(output_data[0])

        intensity = [row[0]-6 for row in input_data]
        floors = [row[1] for row in input_data]
        GM = [row[2] for row in input_data]
        LCol = [row[3] for row in input_data]
        LSpan = [row[4] for row in input_data]
        LBeam1 = [row[5] for row in input_data]
        LBeam2 = [row[6] for row in input_data]
        MsDI = [row[7] for row in input_data]

        intensity_onehot = np_utils.to_categorical(intensity, num_classes=3)
        floors_minMax = normalize(np.array(floors).reshape(-1, 1), 3, 12)
        LCol_minMax = normalize(np.array(LCol).reshape(-1, 1), 3000, 3600)
        LSpan_minMax = normalize(np.array(LSpan).reshape(-1, 1), 3600, 7200)
        LBeam1_minMax = normalize(np.array(LBeam1).reshape(-1, 1), 5400, 6600)
        LBeam2_minMax = normalize(np.array(LBeam2).reshape(-1, 1), 2400, 3600)

        print(intensity_onehot[0])

        output_value = np.array(output_data)
        output_value = np.log(output_value)
        y_Data = output_value
        X_Data = np.column_stack((GM, intensity_onehot, floors_minMax, LCol_minMax, LSpan_minMax, LBeam1_minMax, LBeam2_minMax, MsDI, floors))

        self.test_Mpaths = X_Data
        self.test_labels = y_Data

        print(self.test_labels.shape)
        np.savetxt("./predicts-val/truth.txt", self.test_labels)

class Model_StrInfo_NN:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        # model = load_model(model_path)
        self.model = load_model(model_path, custom_objects={"modify_mae": modify_mae, "modify_rmse": modify_rmse})
        self.model.summary()

    def evaluate(self, dataset, batch_size):
        GM_data = pd.read_excel("AsNames.xlsx")
        GM_data_li = GM_data.values.tolist()
        AsList = []
        for s_li in GM_data_li:
            AsList.append(s_li[0])
        def generator(out):
            a = 1
            test_size = batch_size
            j = 0
            while 1:
                x_Data = dataset.test_Mpaths[j:((j + test_size))]
                y = dataset.test_labels[j:((j + test_size))]
                y = np.array(y)
                j = (j + test_size)
                x_floors = []
                MsDIs = []
                x_GM, x_StrInfo = [], []
                for i, x_path in enumerate(x_Data):
                    As_path = AsList[int(x_path[0])]
                    GMData = np.loadtxt("./AfterShock/" + As_path)
                    x_GM.append(GMData)
                    StrInfo = x_path[1:9]
                    x_StrInfo.append(StrInfo)
                    MsDI = x_path[9]
                    MsDIs.append(MsDI)
                    floors = x_path[10]
                    x_floors.append(floors)
                x_floors = np.array(x_floors)
                x_floors = x_floors[:, np.newaxis]
                y = np.append(y, x_floors, axis=1)
                x_MsDIs = np.array(MsDIs)
                x_GMs = np.array(x_GM)
                x_StrInfos = np.array(x_StrInfo)
                x = [x_MsDIs, x_StrInfos, x_GMs]
                if out:
                    a = a + 1
                if len(y) == 0:
                    break
                yield x, y

        predicts = self.model.predict(generator(out=True),
                                      steps=int((dataset.test_Mpaths.shape[0])/batch_size),
                                      callbacks=None,
                                      max_queue_size=10,
                                      workers=1,
                                      use_multiprocessing=False,
                                      verbose=1)

        np.savetxt("./predicts-val/predicts.txt", predicts, fmt="%s")

if __name__ == '__main__':
    pred_path = "./predicts-val"
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    filepath = 'case5-74MsDIAs.xlsx'
    dataset = Dataset(filepath=filepath)
    dataset.load()

    model_name = "model_54_0.1676"
    batch_size = 1
    model_path = './model/%s.h5' %model_name
    model = Model_StrInfo_NN()
    model.load_model(model_path)
    
    start = time.time()
    model.evaluate(dataset, batch_size)
    end = time.time()
    print(end - start)