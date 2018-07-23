import data_preprocess as dataBuilder
import numpy as np
import pandas as pd
from theano import tensor as T, function, printing
from keras.models import Sequential
#import customLayer as cLayer
from collections import defaultdict
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, Flatten, Embedding, Merge, Dropout
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Bidirectional, SimpleRNN
from keras.layers.merge import Concatenate
from keras.models import Model

import optparse


optparser = optparse.OptionParser()
optparser.add_option(
    #Rel_e1st+ste1
    "-i", "--input", default="inputs/predict.txt",
    help="input text file"
)

optparser.add_option(
    "-m", "--model", default="blstm+cnn",
    help="model structure"
)
opts = optparser.parse_args()[0]
assert os.path.isfile(opts.input)

#D:/DeepLearning_data/w2v/
W2V_PATH = '/mnt/d/DeepLearning_data/w2v/PubMed-and-PMC-w2v.bin'
INPUT_PATH = opts.input

MAX_NB_WORDS = 2000
EMBEDDING_DIM = 300

result = dataBuilder.get_processed_res(W2V_PATH, INPUT_PATH)
EMBEDDING_DIM = dataBuilder.W2V_DIM

print "Embedding dimension: " + str(EMBEDDING_DIM)

'''
co_sent: TP-phenotype co-occurrence sentence
W: w2v matrix[vocab(1716)*w2v dimension(300)]. each row means specific word in vocab.
W2: random vector matrix[vocab(1716)*w2v dimension(300)].
word_idx_map: word index of W dictionary. key : word, value: W's row order.( i-th row ) ex. people : 3
vocab: dictionary; [ word : frequenct(float) ]
'''


def get_idx_from_sent(sent, word_idx_map, max_l=170, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    # x : a list of indices
    x = []
    pad = filter_h - 1

    # pad 0s due to convolution filter size.
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    # up to max length + padding.
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=144, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """

    train_w, train_y = [], []
    for rev in revs:
        # convert a sentence to a list of indices.
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        # sent.append(rev["y"])
        # divide data into test and train.

        train_w.append(sent)
        train_y.append(rev["y"])
    
    print len(train_w[2])
    train_x = np.array(train_w, dtype="int")
    train_t = to_categorical(np.array(train_y, dtype="int"))

    return train_x, train_t


def evaluate(co_sents, word_idx_map, max_l, filter_h, model, nb_epochs = 25, batchs = 50 ):
    x_train, y_train = make_idx_data_cv(co_sents, word_idx_map, 1, max_l, filter_h) # 1 : test set.

    #load trained model.
    model.load_weights("models/model22.h1", by_name=True)
    predictions = model.predict( x_train)
    y_classes = np.argmax(predictions, axis=-1)
    
    for y in y_classes :
    
        fsw.write(str(y)+'\n')


def cv(co_sents, word_idx_map, max_l, filter_h, model, nb_epochs = 25, batchs = 50 ):

    cvscores = []
    r = range(0, 10)
    for i in r:
        x_train, y_train, x_test, y_test = make_idx_data_cv(co_sents, word_idx_map, i, max_l, filter_h)
        print "CV: " + str(i+1)
        print('Traing and validation set number of [Increase, Decrease, Unknown, No-relation] of co-occurrence sentences. ')
        print y_train.sum(axis=0)
        print y_test.sum(axis=0)

        model.load_weights("model.h001")

        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs = nb_epochs, batch_size= batchs)

        scores = model.evaluate(x_test, y_test, verbose=0)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)


    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    fsw.write("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print(cvscores)
    fsw.write(str(cvscores)+'\n')

def rnn_model( co_sents, W, W2, word_idx_map, vocab, epochs = 25, batchs = 50, use_bias=True ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False)

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)
    l_recurrent = Bidirectional( SimpleRNN( 100, activation = "relu",return_sequences=True,recurrent_dropout=0.4 ) )(embedded_sents)
    #l_flat = Flatten()(l_recurrent)
    l_pool = MaxPooling1D(pool_size=(max_l))(l_recurrent)
    l_flat = Flatten()(l_pool)
    l_drop = Dropout(0.5)(l_flat)
    preds = Dense(4, activation='softmax' )(l_drop)
    model = Model(input_sents, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting -RNN")
    model.summary()
    model.save_weights("model.h7")
    cv(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )

def blstm_model( co_sents, W, W2, word_idx_map, vocab, epochs = 25, batchs = 50, use_bias=True ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False)

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)

    l_recurrent = Bidirectional( LSTM(100, recurrent_activation = "tanh",return_sequences=True,recurrent_dropout=0.4 ) )(embedded_sents)
    l_pool = MaxPooling1D(pool_size=(max_l))(l_recurrent)
    l_flat = Flatten()(l_pool)
    l_drop = Dropout(0.5)(l_flat)
    preds = Dense(4, activation='softmax' )(l_drop)
    model = Model(input_sents, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting -BLSTM")
    model.summary()
    model.save_weights("model.h7")
    cv(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )

def blstm_cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epochs = 25, batchs = 50 ):


    embedding_layer = Embedding(9000,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False, name = 'new_embedd')

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)

    l_gru = Bidirectional( LSTM(100, recurrent_activation = "tanh",return_sequences=True, recurrent_dropout = 0.3), name = 'lstm_1')(embedded_sents)

    for fsz in filters:
        # paramaters mean filter tensors shape.
        l_conv = Conv1D( filters=25, kernel_size=fsz,
                        activation="relu",
                        use_bias=True , name = 'conv_'+str(fsz))( l_gru )
        l_pool = MaxPooling1D(pool_size=(max_l - (fsz - 1)), name = 'pool_'+str(fsz))(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1,name='concatenate_1')(convs)
    #l_merge = Merge(concat_axis= 1, mode='concat')(convs)

    l_flat = Flatten(name = 'flatten_1')(l_merge)
    #l_dense = Dense(100, activation='relu')(l_flat)
    l_drop = Dropout(0.5, name = 'dropout_1')(l_flat)
    preds = Dense(4, activation='softmax', name = 'dense_1')(l_drop)
    model = Model(input_sents, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])


    print("model fitting -BLSTM + CNN")
    model.summary()
    #model.save_weights("model.h7")
    #cv(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )
    evaluate(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )

def cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epochs = 25, batchs = 50 ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False)

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)

    for fsz in filters:
        # paramaters mean filter tensors shape.
        l_conv = Conv1D( filters=30, kernel_size=fsz,
                        activation="relu",
                        use_bias=True)( embedded_sents )
        l_pool = MaxPooling1D(pool_size=(max_l - (fsz - 1)))(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    #l_merge = Merge(concat_axis= 1, mode='concat')(convs)


    l_flat = Flatten()(l_merge)
    l_dense = Dense(100, activation='relu')(l_flat)
    l_drop = Dropout(0.5)(l_dense)
    preds = Dense(4, activation='softmax' )(l_drop)
    model = Model(input_sents, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    #rmsprop
    print("model fitting -CNN")
    model.summary()
    model.save_weights("model.h7")
    cv(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs)
    #evaluate(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )


if __name__ == "__main__" :
    co_sents, W, W2, word_idx_map, vocab = result[0], result[1], result[2], result[3], result[4]
    print "data loaded!"

    max_l = 300#np.max(pd.DataFrame(co_sents)["num_words"])
    # for convolution
    filters = [3,5,7]
    filter_h = 7
    pad = filter_h - 1
    convs = []
    #e1_tuple = (  word_idx_map['e1'], word_idx_map['start'], word_idx_map['end'], word_idx_map['e2'] )
    fsw = open("/home/hjb/Dropbox/prid_resl2.txt", 'a')
    #print e1_tuple
    fsw.write( "=============================================="+'\n')
    fsw.write( str(opts.input)+'\n')
    fsw.write( str(opts.model)+'\n')

    batch = [35,50]

    epoch = [23,30,17,25,60]
    print (str(opts.input))
    print ('batch size[lstm+cnn, cnn] : '+ str(batch) )

    if opts.model == 'blstm+cnn':
       
        epoch_x = epoch[3]
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        blstm_cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epoch_x, batch[0] )
    #only reccurent layer.
        
    elif opts.model == 'blstm':
        epoch_x = epoch[1]
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        blstm_model( co_sents, W, W2, word_idx_map, vocab, epoch_x, batch[0] )
    #only cnn layer.
  
    fsw.close();
    # train, validation, test data sets.
