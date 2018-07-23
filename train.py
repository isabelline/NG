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
from keras.layers import Dense, Input, Flatten, Embedding, Merge, Dropout, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Bidirectional, SimpleRNN
from keras.layers.merge import Concatenate
from keras.models import Model
import win_unicode_console
from sklearn.metrics import classification_report
from sklearn import metrics
import optparse
import operator
#from kutilities.layers import AttentionWithContext
from keras.layers.core import *
from keras.models import *
from keras.layers import merge
from keras import backend as K
from keras import initializers
from sklearn.metrics import confusion_matrix



win_unicode_console.enable()

optparser = optparse.OptionParser()
optparser.add_option(
    #Rel_e1st+ste1
    "-i", "--input", default="inputs/reparse.txt",
    help="input text file"
)

optparser.add_option(
    "-m", "--model", default="blstm+cnn",
    help="model structure"
)
opts = optparser.parse_args()[0]
assert os.path.isfile(opts.input)

#D:/DeepLearning_data/w2v/
#W2V_PATH = 'D:/datastories.twitter.200d.txt'
W2V_PATH = 'C:/Users/admin/Documents/PubMed-and-PMC-w2v.txt'

#INPUT_PATH = "C:/Users/admin/Documents/sst5_train_sentences.txt"
INPUT_PATH = "inputs/reparse.txt"
NEW_PATH = "inputs/app.txt"

MAX_NB_WORDS = 300
EMBEDDING_DIM = 200

result = dataBuilder.get_processed_res(W2V_PATH, INPUT_PATH)
result2 = dataBuilder.get_processed_res2(W2V_PATH, NEW_PATH)
EMBEDDING_DIM = dataBuilder.W2V_DIM

print ("Embedding dimension: " + str(EMBEDDING_DIM))

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
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    # up to max length + padding.

    while len(x) < max_l + 2 * pad:
        x.append(0)
        
    return x




def get_idx_from_sent2(sent, word_idx_map, max_l=170, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    # x : a list of indices
    x = []
    pad = filter_h - 1

    # pad 0s due to convolution filter size.
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    # up to max length + padding.

    if len(x) >= max_l + 2 * pad:
        return x[0:max_l+2*pad]
    else:
        while len(x) < max_l + 2 * pad:
            x.append(0)
        return x






def make_idx_data_cv2(revs, word_idx_map, cv, max_l=144, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """

    train_w, train_y, test_w, test_y = [], [], [], []
    for rev in revs:
        # convert a sentence to a list of indices.
        sent = get_idx_from_sent2(rev["text"], word_idx_map, max_l, filter_h)
        # sent.append(rev["y"])
        # divide data into test and train.
   #     if rev["split"] == cv:
   #         test_w.append(sent)
   #         test_y.append(rev["y"])
   #     else:
   #         train_w.append(sent)
   #        train_y.append(rev["y"])

        train_w.append(sent)
        train_y.append(rev["y"])


    train_w = np.array(train_w, dtype="int")
    train_y = to_categorical(np.array(train_y, dtype="int"))
  #  test_w = np.array(test_w, dtype="int")
  #  test_y = to_categorical(np.array(test_y, dtype="int"))

    return train_w, train_y, [], []






def make_idx_data_cv(revs, word_idx_map, cv, max_l=144, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """

    train_w, train_y, test_w, test_y = [], [], [], []
    for rev in revs:
        # convert a sentence to a list of indices.
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        # sent.append(rev["y"])
        # divide data into test and train.
        if rev["split"] == cv:
            test_w.append(sent)
            test_y.append(rev["y"])
        else:
            train_w.append(sent)
            train_y.append(rev["y"])

    train_w = np.array(train_w, dtype="int")
    train_y = to_categorical(np.array(train_y, dtype="int"))
    test_w = np.array(test_w, dtype="int")
    test_y = to_categorical(np.array(test_y, dtype="int"))

    return train_w, train_y, test_w, test_y










def evaluate(co_sents, word_idx_map, max_l, filter_h, model, nb_epochs = 25, batchs = 50 ):
    x_train, y_train, x_test, y_test = make_idx_data_cv(co_sents, word_idx_map, 1, max_l, filter_h) # 1 : test set.

    print('Traing and validation set number of [Increase, Decrease, Unknown, No-relation] of co-occurrence sentences. ')
    print (y_train.sum(axis=0))
    print (y_test.sum(axis=0))

    #model.load_weights("model.h7")
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs = nb_epochs, batch_size= batchs)

    scores = model.evaluate(x_test, y_test, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3] * 100))
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
    
    fsw.write("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def get_label_category(test_x):
    predict_label = []
    for line in test_x:
        index, value = max(enumerate(line), key=operator.itemgetter(1))
        predict_label.append(index)
    return predict_label


def print_metrics(predict_y, test_y):
    true_label = []
    for line in test_y:
        index, value = max(enumerate(line), key=operator.itemgetter(1))
        true_label.append(index)

    target_names = ['increase', 'decrease', 'unknown', 'no-relation']
    print(classification_report(true_label, predict_y, target_names=target_names))
    print(confusion_matrix(true_label, predict_y), target_names = target_names)
    print("weighted f1 score: ")
    print(metrics.f1_score(true_label, predict_y, average ='weighted'))

def cv(co_sents, co_sents2, word_idx_map, max_l, filter_h, model, nb_epochs = 25, batchs = 50 ):

    cvscores = []
    r = range(0, 10)
    for i in r:
        x_train, y_train, x_test, y_test = make_idx_data_cv(co_sents, word_idx_map, i, max_l, filter_h)
       # x_train2, y_train2, x_test2, y_test2 = make_idx_data_cv2(co_sents2, word_idx_map, i, max_l, filter_h)
       # print("x_train2 len " + str(len(x_train2)))
       # print("y_train2 len " + str(len(y_train2)))
       # print("x_test2 len " + str(len(x_test2)))
       # print("y_test2 len " + str(len(y_test2)))
        print ("CV: " + str(i+1))
        print('Traing and validation set number of [Increase, Decrease, Unknown, No-relation] of co-occurrence sentences. ')
        print (y_train.sum(axis=0))
        print (y_test.sum(axis=0))

      #  model.load_weights("model.h7")
       # K.set_learning_phase(1)

        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs = nb_epochs, batch_size= batchs)

        '''
        predict class and get precision and recall
        '''

        prob_category = model.predict(x_test, verbose = 1)
       # prob_category_2 = model.predict(x_train2, verbose = 1)
        label_category = get_label_category(prob_category)
    #    label_category_2 = get_label_category(prob_category_2)

     #   f = open("predict_55_" + str(i)+ ".txt", 'w')
     #   print("label len " + str(len(label_category_2)))

     #   for line in label_category_2:
     #       f.write(str(line)+'\n')


        print_metrics(label_category, y_test)



        scores = model.evaluate(x_test, y_test, verbose=0)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)


    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    fsw.write("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print(cvscores)
    fsw.write(str(cvscores)+'\n')
    if np.mean(cvscores) > 80.0 :
        model.save('model22.h1')













def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = 312
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model
































#def position_embedding( co_sents, word_idx_map ):
#    words = co_sents["text"];
def lstm_model( co_sents, W, W2, word_idx_map, vocab, epochs = 25, batchs = 50 , use_bias=True):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False)

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)

    l_recurrent = LSTM(100, recurrent_activation = "relu", return_sequences=True)(embedded_sents)
    #l_flat = Flatten()(l_recurrent)
    l_pool = MaxPooling1D(pool_size=(max_l))(l_recurrent)
    l_flat = Flatten()(l_pool)
    preds = Dense(4, activation='softmax' )(l_flat)
    model = Model(input_sents, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting -LSTM")
    model.summary()
    model.save_weights("model.h7")
    cv(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )

def rnn_model( co_sents, W, W2, word_idx_map, vocab, epochs = 25, batchs = 50, use_bias=True ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False)

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)
    l_recurrent = Bidirectional( SimpleRNN( 100, activation = "relu",return_sequences=True,recurrent_dropout=0.4 ), merge_mode = 'sum'  )(embedded_sents)
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

def blstm_model( co_sents, W, W2, word_idx_map, vocab, epochs = 3, batchs = 50, use_bias=True ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False , name = "embedd_l" )

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)
#recurrent_dropout=0.4 recurrent_activation = "tanh",
    l_recurrent = Bidirectional( LSTM(100, return_sequences=True,recurrent_dropout=0.4, recurrent_activation = "tanh" ),merge_mode = 'sum' )(embedded_sents)
   #pool_size=(max_l)
  #  attention_mul = attention_3d_block(l_recurrent)

    l_pool = MaxPooling1D()(l_recurrent)
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







def blstm_cnn_model( co_sents, co_sents2, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epochs = 25, batchs = 50 ):


    embedding_layer = Embedding(len(word_idx_map)+1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False, name = 'embedd_1')

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='int32')
    embedded_sents = embedding_layer(input_sents)
#, recurrent_dropout = 0.3
    l_gru = Bidirectional( LSTM(100, recurrent_activation = "tanh",return_sequences=True), name = 'lstm_1')(embedded_sents)
 #   l_lstm = Bidirectional( LSTM(100, recurrent_activation = "tanh",return_sequences=True), name = 'lstm_1')(embedded_sequences)

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

   # l_drop.add(AttentionWithContext())

    preds = Dense(4, activation='softmax', name = 'dense_1')(l_drop)
    model = Model(input_sents, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting -BLSTM + CNN")
    model.summary()
 #   model.save_weights("model.h7")
    cv(co_sents, co_sents2, word_idx_map, max_l, filter_h, model, epochs, batchs )
    #evaluate(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs )


def cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epochs = 25, batchs = 50 ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                                EMBEDDING_DIM,
                                weights=[W],
                                input_length = (max_l + 2 * pad),
                                trainable = False)

    input_sents = Input(shape=(max_l + 2 * pad,), dtype='float32')

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

def word_bench( vocab ) :
    word10 = defaultdict(float)
    word50 = defaultdict(float)
    word100 = defaultdict(float)
    word500 = defaultdict(float)

    for k, v in vocab.items() :

        if vocab[k] >= 10 and vocab[k] < 20 :
            word10[k] = v;

        if vocab[k] >= 50 and vocab[k] < 60 :
            word50[k] = v;

        if vocab[k] >= 100 and vocab[k] < 150 :
            word100[k] = v;

        if vocab[k] >= 500 and vocab[k] < 550 :
            word500[k] = v;

'''
        print str(word10)
        print str(word50)
        print str(word100)
        print str(word500)

'''

class MyLayer(Layer):
	def __init__(self, **kwargs):
		super(MyLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		print("333333333333333333")
		print(input_shape)
		self.kernel = self.add_weight(name = 'kernel',
			shape = (input_shape[0], input_shape[-1]),
			initializer = 'uniform',
			trainable=True)
		super(MyLayer, self).build(input_shape)

	def call(self, x):
		return K.dot(x, self.kernel)

	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[-1]


class Attention2(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
       #\
  #      self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.l2(0.01)
        #self.W_regularizer = regularizers.get(W_regularizer)
    
        self.b_regularizer = regularizers.l2(0.01)
        self.W_constraint = constraints.max_norm(2.)
        self.b_constraint = constraints.max_norm(2.)

        self.bias = bias
        super(Attention2, self).__init__(**kwargs)

    def build(self, input_shape):
    	assert len(input_shape) == 3
    	self.W = self.add_weight((input_shape[-1],),initializer=self.init,name='W', regularizer=self.W_regularizer, constraint=self.W_constraint)
    	self.b = self.add_weight((input_shape[1],), initializer='zero', name='b',regularizer=self.b_regularizer, constraint=self.b_constraint)
    	self.built = True
   # 	self.input_spec = InputSpec(min_ndim = 2, axes={-1:input_shape[-1]})
    #	super(Attention2, self).build(input_shape)



    def call(self, x, mask=None):
        eij = K.dot(x, self.W)
        eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
    	print((input_shape[0], input_shape[-1]))
    	return (input_shape[0], input_shape[-1])

    def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])



def attention( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epochs = 25, batchs = 50 ):

    embedding_layer = Embedding(len(word_idx_map) + 1,
                            EMBEDDING_DIM,
                            weights=[W],
                            input_length=(max_l + 2 * pad),
                            trainable=False)
# input = (batch_size, array_of_word_index)
# output = (batch_size, max_sentence_length + padding, word2vec_dim)
 
#why need this?  
    sentence_input = Input(shape=(max_l + 2 * pad,), dtype='float32')
    #input = (batch, sentence_length,)
    #output = same as input (??)

    #preds = sentence_input

    
 #   print("--------------------")
#    print(sentence_input[0].eval())

    embedded_sequences = embedding_layer(sentence_input)
    #size = (batch_size, max_sentence_length + padding, word2vec_dim)
 #recurrent_dropout = 0.3
    l_lstm = Bidirectional( LSTM(100, recurrent_activation = "tanh",return_sequences=True), name = 'lstm_1')(embedded_sequences)
    #input =(batch_size, max_sentence_length + padding, word2vec_dim)
    #size = (batch_size, 100)

    l_att = Attention2()(l_lstm)
  #  l_flat = Flatten()(l_lstm)

    preds = Dense(4, activation='softmax')(l_att)
    
    model = Model(sentence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
#    model.save_weights("model.h7", overwrite = True)
  #model.set_weights()
    cv(co_sents, word_idx_map, max_l, filter_h, model, epochs, batchs)


    #input (batch, sentence_length)
	#output = (batch_size, max_sentence_length + padding, word2vec_dim)

'''
    l_recurrent = Bidirectional( LSTM(100, return_sequences=True,recurrent_dropout=0.4, recurrent_activation = "tanh" ),merge_mode = 'sum' )(embedded_sequences)
   #pool_size=(max_l)
  #  attention_mul = attention_3d_block(l_recurrent)

    l_pool = MaxPooling1D()(l_recurrent)
    l_flat = Flatten()(l_pool)

 

     #GRU(100, return_sequences=True)
 #   l_lstm = Bidirectional( LSTM(100, recurrent_activation = "tanh",return_sequences=True, recurrent_dropout = 0.3), name = 'lstm_1')(embedded_sequences)
 #   l_dense = TimeDistributed(Dense(200))(embedded_sequences)

    l_my = MyLayer()(l_flat)

  #  l_att = AttLayer()(l_dense)

  
    preds = Dense(4, activation='softmax')(l_my)
    '''
    #Model (layer, layer)
   






if __name__ == "__main__" :
    #
    co_sents, W, W2, word_idx_map, vocab = result[0], result[1], result[2], result[3], result[4]
    co_sents2, W2, W22, word_idx_map2, vocab2 = result2[0], result2[1], result2[2], result2[3], result2[4]

    print ("data loaded!")

    max_l = 300 #  calculte max sentence length : np.max(pd.DataFrame(co_sents)["num_words"])
    # for convolution
    filters = [3,5,7]
    filter_h = 7
    pad = filter_h - 1
    convs = []
    
    # file out; final output( an average of CV)
    fsw = open("test.txt", 'a')
    fsw.write( "=============================================="+'\n')
    fsw.write( str(opts.input)+'\n')
    fsw.write( str(opts.model)+'\n')

    batch = [35,50]

    epoch = [23,27,17,25,60]
    print (str(opts.input))
    print ('batch size[lstm+cnn, cnn] : '+ str(batch) )
    print(opts.model)

    if opts.model == 'lstm+cnn' :
        epoch_x = epoch[0]
        print ('epoch size : '+ str( epoch_x ) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        lstm_cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epoch_x, batch[0] )
    #cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad )
    elif opts.model == 'blstm+cnn':
      #  epoch_x = 1
        epoch_x =27
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        blstm_cnn_model( co_sents, co_sents2, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epoch_x, batch[0] )
    #only reccurent layer.
    elif opts.model == 'lstm':
        epoch_x = epoch[1]
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        lstm_model( co_sents, W, W2, word_idx_map, vocab, 27, batch[0] )
        
    elif opts.model == 'blstm':
        epoch_x = 40
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        blstm_model( co_sents, W, W2, word_idx_map, vocab, epoch_x, batch[0] )
    #only cnn layer.
    elif opts.model == 'rnn':
        epoch_x = epoch[4]
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        print ('epoch size : '+ str(epoch_x) )
        rnn_model( co_sents, W, W2, word_idx_map, vocab, epoch_x, batch[0] )     
    elif opts.model == 'cnn':
        epoch_x = 60
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        cnn_model( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epoch_x, batch[1] )
    elif opts.model == 'att':
        epoch_x = 27
        print ('epoch size : '+ str(epoch_x) )
        fsw.write('epoch size : '+ str(epoch_x)+ '\n');
        attention( co_sents, W, W2, word_idx_map, vocab, convs,max_l, filters, filter_h, pad, epoch_x, batch[1] )

    fsw.close();
    # train, validation, test data sets.

