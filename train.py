import os
import time
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Activation
from keras.backend import set_session
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from keras.layers import Input, concatenate, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.mae = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_mae = {'batch': [], 'epoch': []}

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.mae['epoch'].append(logs.get('modify_mae'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_mae['epoch'].append(logs.get('val_modify_mae'))

    def loss_plot(self, loss_type,model_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'k', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'r', label='val loss')
        plt.title(model_name)
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig("./result_data/%sloss.png"%(model_name))
    def mae_plot(self, mae_type,model_name):
        iters = range(len(self.mae[mae_type]))
        plt.figure()
        plt.plot(iters, self.mae[mae_type], 'k', label='train mae')
        if mae_type == 'epoch':
            plt.plot(iters, self.val_mae[mae_type], 'r', label='val mae')
        plt.title(model_name)
        plt.grid(True)
        plt.xlabel(mae_type)
        plt.ylabel('mae')
        plt.legend(loc="upper right")
        plt.savefig("./result_data/%smae.png"%(model_name))

        np.savetxt("./result_data/%s_train_mae_epoch.txt" %(model_name),self.mae['epoch'],fmt="%s")
        np.savetxt("./result_data/%s_train_loss_epoch.txt" %(model_name),self.losses['epoch'],fmt="%s")
        np.savetxt("./result_data/%s_test_mae_epoch.txt" %(model_name),self.val_mae['epoch'],fmt="%s")
        np.savetxt("./result_data/%s_test_loss_epoch.txt" %(model_name),self.val_loss['epoch'],fmt="%s")

        # plt.show()
def normalize(values, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in values]
class Dataset:
    def __init__(self, filepath):
        self.file_path = filepath
    def train_load(self):
        r_data = pd.read_excel(self.file_path)
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

        output_value = np.array(output_data)
        output_value = np.log(output_value)

        X_Data = np.column_stack((GM, intensity_onehot, floors_minMax, LCol_minMax, LSpan_minMax, LBeam1_minMax, LBeam2_minMax, MsDI, floors))
        print(X_Data[0])
        y = output_value
        X_train, X_temp, y_train, y_temp = train_test_split(X_Data, y, test_size=0.2, random_state=0)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

        GM_data = pd.read_excel("./AsNames.xlsx")
        GM_data_li = GM_data.values.tolist()
        AsList = []
        for s_li in GM_data_li:
            AsList.append(s_li[0])
        print(len(AsList))

        y = np.array(y_train)
        x_floors = []
        MsDIs = []
        x_GM, x_StrInfo = [], []
        for i, x_path in enumerate(X_train):
            As_path = AsList[int(x_path[0])]
            GMData = np.loadtxt("./AfterShock/" + As_path)
            x_GM.append(GMData)
            StrInfo = x_path[1:9]
            x_StrInfo.append(StrInfo)

            MsDI = x_path[9]
            MsDIs.append(MsDI)
            floors = x_path[10]
            x_floors.append(floors)
            print(i)
        x_floors = np.array(x_floors)
        x_floors = x_floors[:, np.newaxis]
        y = np.append(y, x_floors, axis=1)
        x_MsDIs = np.array(MsDIs)
        x_GMs = np.array(x_GM)
        x_StrInfos = np.array(x_StrInfo)
        x = [x_MsDIs, x_StrInfos, x_GMs]

        self.train_Mpaths = x
        self.train_labels = y
        print(self.train_labels.shape)

        y = np.array(y_val)
        x_floors = []
        MsDIs = []
        x_GM, x_StrInfo = [], []
        for i, x_path in enumerate(X_val):
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

        self.val_Mpaths = x
        self.val_labels = y

        print(self.val_labels.shape)

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


class Model_StrInfo_NN:
    def __init__(self):
        self.StrInfo_NN = None

    def build_model(self):
        self.GM_CNN = Sequential()

        self.GM_CNN.add(Conv1D(8, kernel_size=3, padding='same', input_shape=(3000, 1)))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(16, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(32, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(64, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(128, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(256, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(256, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Conv1D(256, kernel_size=3, padding='same'))
        self.GM_CNN.add(Activation('relu'))
        self.GM_CNN.add(MaxPooling1D(pool_size=2, strides=2))

        self.GM_CNN.add(Flatten())
        self.GM_CNN.add(Dense(128))
        self.GM_CNN.add(Activation('relu'))

        self.GM_CNN.add(Dense(8))
        self.GM_CNN.add(Activation('relu'))

        StruInfo_input= Input(shape=(8,))
        MsDI_input= Input(shape=(1,))

        GM_CNN_input = Input(shape=(3000, 1))
        GM_CNN_out = self.GM_CNN(GM_CNN_input)

        concatenated_input = concatenate([MsDI_input, StruInfo_input, GM_CNN_out])
        out = Dense(128, activation='relu')(concatenated_input)
        out = Dense(128, activation='relu')(out)
        output = Dense(14)(out)
        self.model = Sequential()
        self.model = Model(inputs=[MsDI_input, StruInfo_input, GM_CNN_input], outputs=output)

    def train(self, dataset, batch_size, nb_epoch, history, model_name):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        set_session(tf.compat.v1.Session(config=config))
        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, amsgrad=True)

        model_path="./model/%s_" %model_name
        filename=model_path+"{epoch:02d}_{val_loss:.4f}.h5"

        checkpoint=ModelCheckpoint(filepath=filename,monitor="val_loss",mode="min",
                                   save_weights_only=False,save_best_only=False,verbose=1,period=1)
        callback_lists=[history,checkpoint]
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=5, restore_best_weights=True)

        self.model.compile(loss=modify_rmse, optimizer=adam, metrics=modify_mae)
        self.model.fit(x=dataset.train_Mpaths,
                       y=dataset.train_labels,
                       batch_size=batch_size,
                       epochs=nb_epoch,
                       callbacks=[early_stopping, callback_lists],
                       validation_data=(dataset.val_Mpaths, dataset.val_labels),
                       validation_batch_size=batch_size)

if __name__=='__main__':
    if not os.path.exists("./result_data"):
        os.mkdir("./result_data")
    if not os.path.exists("./model"):
        os.mkdir("./model")

    filepath = '../200MsDIAsDI.xlsx'
    dataset = Dataset(filepath=filepath)
    dataset.train_load()

    model = Model_StrInfo_NN()
    model.build_model()
    model_name = "model"
    history = LossHistory()
    batch_size, nb_epoch = 512, 100
    start = time.time()
    model.train(dataset, batch_size, nb_epoch, history, model_name=model_name)
    end = time.time()
    print("总耗时=")
    print(end-start)
    history.loss_plot('epoch', model_name=model_name)
    history.mae_plot('epoch', model_name=model_name)

