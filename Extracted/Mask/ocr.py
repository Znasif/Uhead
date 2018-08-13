import keras
import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)

import os
from os.path import join
import json
import random
import itertools
import re
import datetime

# import editdistance
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model, model_from_yaml
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import cv2

sess = tf.Session()
K.set_session(sess)

direc = os.getcwd()

from collections import Counter



letters_train = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

letters = sorted(list(letters_train))
print('Letters:', ' '.join(letters))


######################################## ---------------------->>> Input Data generator

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True


class TextImageGenerator:

    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=1
                 ):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.samples = []

        ################..------------------------------------------------>> changes has been made

        fin = 0
        for dir in os.listdir(dirpath):
            cnt = 0
            img_dirpath = join(dirpath, dir)
            for filename in os.listdir(img_dirpath):
                cnt += 1
                fin += 1
                # if cnt > 50:
                #   break

                name, ext = os.path.splitext(filename)
                if ext == '.jpg' or '.png' or '.tif':
                    img_filepath = join(img_dirpath, filename)
                    description = dir
                    if is_valid_str(description):
                        self.samples.append([img_filepath, description])

            self.n = len(self.samples)
            self.indexes = list(range(self.n))
            self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        self.direcs = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.bitwise_not(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)
            self.direcs.append(img_filepath)

    def get_output_size(self):
        return len(letters) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        indx = self.indexes[self.cur_index]
        return self.imgs[indx], self.texts[indx], self.direcs[indx]

    def all_data(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.n, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.n, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.n])
            input_length = np.ones((self.n, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.n, 1))
            source_str = []
            direcl = []

            for i in range(self.n):
                img, text, direc = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                direcl.append(direc)
                Y_data[i] = text
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                'direc' : direcl
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.n])}
            yield (inputs, outputs)

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
            direc = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


##################----------------------->>>>>>>>>   END


'''
for inp, out in tiger.next_batch():

    print('Text generator output (data which will be fed into the neutral network):')
    print('1) the_input (image)')
    if K.image_data_format() == 'channels_first':
        img = inp['the_input'][0, 0, :, :]
    else:
        img = inp['the_input'][0, :, :, 0]

    plt.imshow(img.T, cmap='gray')
    plt.show()

    print('2) the_labels (plate number): %s is encoded as 12233' %
          (labels_to_text(inp['the_labels'])))
    #print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
          #(inp['input_length'][0], tiger.img_w))
    print('4) label_length (length of plate number): %d' % inp['label_length'][0])
    break

#####################################----------------->>> Loss and train functions, network architecture

'''


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(img_w, load=False):
    # Input Parameters
    print("Please wait for a while-----Traning is in process!!!")

    img_h = 28

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    batch_size = 32
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator(join(direc, 'train/'), img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()

    tiger_val = TextImageGenerator(join(direc, 'vald'), img_w, img_h, batch_size, downsample_factor)
    tiger_val.build_data()

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner=Dropout(0.2,name='drop1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = Dropout(0.2, name='drop2')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv3')(inner)

    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))

    y_pred = Activation('softmax', name='softmax')(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    if not load:
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        model.fit_generator(generator=tiger_train.next_batch(),
                            steps_per_epoch=tiger_train.n/32,
                            epochs=32,
                            validation_data=tiger_val.next_batch(),
                            validation_steps=tiger_val.n)

    return model


#####################---------------------- Loss and train functions, network architecture¶-----END

##################-----------------------Training the model
'''
model = train(32, load=False)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''
# later...

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


################------------------------------Function to decode neural network output

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(out):
    ret = []

    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)

    return ret


#########################------------------------END

#########################------------------------Test on validation images

direc1 = "../"
tiger_test = TextImageGenerator(join(direc1, 'ALL'), 32, 28, 32, 4)
tiger_test.build_data()

net_inp = loaded_model.get_layer(name='the_input').input
net_out =  loaded_model.get_layer(name='softmax').output

correct = 0
wrong = 0

for inp_value, _ in tiger_test.all_data():
    bs = inp_value['the_input'].shape[0]
    X_data = inp_value['the_input']
    direc = inp_value['direc']
    net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})

    pred_texts = decode_batch(net_out_value)
    labels = inp_value['the_labels']
    texts = []
    for label in labels:
        text = label
        texts.append(text)

    for i in range(bs):
        '''
        fig = plt.figure(figsize=(10, 10))
        outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
        ax1 = plt.Subplot(fig, outer[0])
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, outer[1])
        fig.add_subplot(ax2)
        '''
        if pred_texts[i] and not pred_texts[i].isspace():
            dem = 4
        else:
            print("MARA")
            continue

        print('FILE: %s\nPredicted: %s\nTrue: %s' % (direc[i], pred_texts[i], int(texts[i])))
        if (int(pred_texts[i]) == int(texts[i])):
            correct += 1
        else:
            wrong += 1

        '''
        img = X_data[i][:, :, 0].T
        ax1.set_title('Input img')
        ax1.imshow(img, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_title('Acrtivations')
        ax2.imshow(net_out_value[i].T, cmap='binary', interpolation='nearest')
        ax2.set_yticks(list(range(len(letters) + 1)))
        ax2.set_yticklabels(letters + ['blank'])
        ax2.grid(False)
        for h in np.arange(-0.5, len(letters) + 1 + 0.5, 1):
            ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

        #ax.axvline(x, linestyle='--', color='k')
        plt.show()
        '''
    break

print(correct)
print(wrong)
print(correct*100 / (correct + wrong))
