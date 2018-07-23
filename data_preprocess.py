import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd
import io

W2V_DIM = 200

def build_data_cv(file, label_dict, cv=10, clean_string=False):
     revs = []
    
    #cnt = 0
     f = open(file, encoding = 'utf-8')
    # defaultdict() if not key-value is valid, return default value(In this case 0.0). 
    # faster than dict.set_default.  
     vocab = defaultdict(float)

     for index, line in enumerate(f.readlines()):   
     	 line.encode('utf-8')
     	 rev = []
     	 rev.append(line.strip())
     	 if clean_string:
     	 	orig_rev = clean_str(" ".join(rev))# trimming special chracters.
     	 else:
     	 	orig_rev = " ".join(rev)

     	 words = set(orig_rev.split())
     	 for word in words:
     	 	vocab[word] += 1 # dictionary; [ word : frequenct(float) ]

         #random split
        

     	 split = np.random.randint( 0, cv )

     	 datum  = {"y":label_dict[index], "text": orig_rev, "num_words": len( orig_rev.split() ),"split": split } # 0~9 set up which cross validation set
     	 revs.append(datum)
        
     f.close;
    
     return revs, vocab
     """
    Loads data and split data( cv )
    """
     




















'''
    """
    Loads data and split data( cv )
    """

    revs = []
    i=0
    vocab = defaultdict(float)
    with open(file, "r") as f:
        for index, line in enumerate(f.readlines()):
            i+=1
            line=line.split('\t')
            y = line[1][:-1]


            clean_rev = clean_str(line[0]).strip()
         #   words = set(clean_rev.split())        
         #   clean_rev = elongated_word(clean_rev)
         #   clean_rev = clean_rev.lower()
            
         #   clean_rev = tknzr.tokenize(clean_rev)    
         #   clean_rev = " ".join(clean_rev)

           
            if y=='very pos':
                y=0
            elif y=='pos':
                y=0
            elif y=='neu':
                continue
            elif y=='neg':
                y=1
            elif y=='very neg':
                y=1
            else:
                print ('class error')
                continue
            

            if clean_rev!="":
                words = set(clean_rev.split())
                for word in words:
                    vocab[word] += 1

             #random split
                split = np.random.randint( 0, cv )

                datum  = {"y":y, 
                          "text": clean_rev,                             
                          "num_words": len(clean_rev.split()),
                          "split":split}                          
                revs.append(datum)   


    return revs, vocab
'''




def get_W(word_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = 6851
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, 200 ), dtype='float32')            
    W[0] = np.zeros( W2V_DIM, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    #fw: for check pretraind word list.
    #fw = open("/home/hjb/Dropbox/w2v_list_pmc.txt", 'w')
    '''
    word_vecs = {}
    with open(fname, "rt") as f:
        header = f.readline()
        print (header)
        #vocab_size, layer1_size = map(int, header.split()) # vocab_size : w2v total vocab size.
        
        global W2V_DIM
        W2V_DIM = 300
        #print ("w2v: "+ str(vocab_size) + " x " + str(W2V_DIM )  )      
        
        binary_len = np.dtype('float32').itemsize * layer1_size # for a word, get word vector values as much as layer1_size.
        for line in range(vocab_size):# split lines by a word.
            word = [] #total words in w2v.bin file.
            while True:
                ch = f.read(1)#read a character
                if ch == ' ': # word is end.
                    word = ''.join(word) # chararcters merge into a word.
                    break
                if ch != '\n':# newline character is not included in a word.
                    word.append(ch)
            #word != 'e1' and word != 'e2' and
            print(word)
            if word in vocab :
            #if word in vocab: #get word vector values in our vocab dictionary.
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               #if( word == 'ett1') : print word
               #if( word == 'ett2') : print word
               
            else:
                f.read(binary_len) #move file pointer next word.
    #fw.close()

    return word_vecs
    '''
    word_vecs = {}
   # vocab_size =400000
   # f =open(fname, "rt", encoding='utf-8')
    '''
    '''
    f =open(fname, "rt", encoding='utf-8')
    
    header = f.readline()
   # vocab_size, layer1_size = map(float, header.split()) # vocab_size : w2v total vocab size.
    
    layer1_size = 200
    vocab_size = len(f.readlines())

    global W2V_DIM
    W2V_DIM = layer1_size
    print ("w2v: "+ str(vocab_size) + " x " + str(W2V_DIM ) )       
        
    binary_len = np.dtype('float32').itemsize * layer1_size # for a word, get word vector values as much as layer1_size.
    f.seek(0)
    '''
        '''
    for i, line in enumerate(f):# split lines by a word.
      #  print(line)
           # word = [] #total words in w2v.bin file.
        line = line.split(' ')
        word = line[0] 
       # print(word)



        if word in vocab :
         #   print("KKKK")
            #print(word)
            #if word in vocab: #get word vector values in our vocab dictionary.
            temp = line[1:201]
            temp = list(map(float, temp))
            word_vecs[word] = np.array(temp)
     #       print (type(word_vecs[word][100]))
    #        print(word_vecs[word])
               #if( word == 'ett1') : print word
               #if( word == 'ett2') : print word
               
        #    else:
        #        f.read(binary_len) #move file pointer next word.
    #fw.close()
    f.close()
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,W2V_DIM )  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def get_labels(file):
    '''
    make prediction label(= y) dictionary.
    i-th sentence's label ; key: i, label: 0~3  
    '''
    f = open(file, encoding='utf-8')
    dict = {}
    for index,label in enumerate(f.readlines()):
        #index,l = i.strip().split('|')
        #index = int(index)
        dict[index] = int(label)
    #    print (dict[0])
    return dict


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()



def get_processed_res( w2v_path, input_path ):
    w2v_file = w2v_path
    #Rela_entityStEn
    #Rela_e12
    #Rela_sym
    #Rela_StEd
    sent_file = input_path # 1_ 10717
    label_file = "inputs/labs.txt"
    #dictionary
    label_dict = get_labels(label_file)
  #  label_dict = {}

#    data_folder = ["rt-polarity.pos","rt-polarity.neg"]
    #get input data
    print ("loading data...")
    revs, vocab = build_data_cv( sent_file, label_dict, cv=10, clean_string=True )
    max_l = np.max(pd.DataFrame(revs)["num_words"]) # check max length.
    print ("data loaded!")
    print ("number of sentences: " + str(len(revs)))
    print ("vocab size: " + str(len(vocab)))
    print ("max sentence length: " + str(max_l))
    print ("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab) # give w2v to our words from w2v.bin 
    print ("word2vec loaded!")
    print ("num words already in word2vec: " + str(len(w2v)))
    # if not a word is in w2v.bin, it is given random vector.
    add_unknown_words(w2v, vocab)
   
    '''
    make W, word_idx_map matix
    W: w2v matrix. [vocab + 1(0th row) * w2v dimention] size matrix. W[0] = zero vector.
    word_idx_map: word and W's index dictionary. key : word, value: W's row order.( i-th row ) ex. people : 3
    '''
    W, word_idx_map = get_W(w2v)
   # W2 : random vector embedding. Do not use w2v.
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    result = [revs, W, W2, word_idx_map, vocab]

    return result 


def get_processed_res2( w2v_path, input_path):
    w2v_file = w2v_path
    #Rela_entityStEn
    #Rela_e12
    #Rela_sym
    #Rela_StEd
    sent_file = input_path # 1_ 10717
    label_file = "inputs/label_all.txt"
    #dictionary
    label_dict = get_labels(label_file)
  #  label_dict = {}

#    data_folder = ["rt-polarity.pos","rt-polarity.neg"]
    #get input data
    print ("loading data...")
    revs, vocab = build_data_cv( sent_file, label_dict, cv=10, clean_string=True )
    max_l = np.max(pd.DataFrame(revs)["num_words"]) # check max length.
    print ("data loaded!")
    print ("number of sentences: " + str(len(revs)))
    print ("vocab size: " + str(len(vocab)))
    print ("max sentence length: " + str(max_l))
    print ("loading word2vec vectors...")
    #w2v = load_bin_vec(w2v_file, vocab) # give w2v to our words from w2v.bin 
    #print ("word2vec loaded!")
    #print ("num words already in word2vec: " + str(len(w2v)))
    # if not a word is in w2v.bin, it is given random vector.
    #add_unknown_words(w2v, vocab)
   
    '''
    make W, word_idx_map matix
    W: w2v matrix. [vocab + 1(0th row) * w2v dimention] size matrix. W[0] = zero vector.
    word_idx_map: word and W's index dictionary. key : word, value: W's row order.( i-th row ) ex. people : 3
    '''
    #W, word_idx_map = get_W(w2v)
   # W2 : random vector embedding. Do not use w2v.
    rand_vecs = {}
    #add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    result = [revs, [], [], [], []]

    return result 