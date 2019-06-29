from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img , img_to_array
from keras.models import Model
from os import listdir
from pickle import dump
import tensorflow as tf
import pandas as pd
from nltk.corpus import stopwords

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

model = VGG16()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

def feature_extraction(img_name):
        
    image = load_img(img_name, target_size =(224,224))
        
    image = img_to_array(image)
        
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
    image = preprocess_input(image)
        
    feature = model.predict(image)
        
    print( img_name , "done!.")
        
    return feature

directory = "E:\\DATA\\images"

features = {}

for name in listdir(directory):
    if name.endswith('jpg'):    
        img_name = directory + "\\" + name
        features[name] = feature_extraction(img_name)
        
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))  

df = pd.read_csv('train.csv')
captions = df['Captions']
image = df['Img_name']

import re
def clean_descriptions(des):
    des = des.lower()
    des = re.sub('[^a-zA-z1-9]', ' ' ,des)
    des = des.split()
    des = [word for word in des if word not in set(stopwords.words())]
    des = ' '.join(des)
    print(des, ">...done")
    return des

descriptions = [clean_descriptions(i) for i in captions]

descriptions = ['startseq'+" "+ i +" " + 'endseq' for i in descriptions ]

f = open('descriptions.txt', 'w')
for i in descriptions:
    string = i +'\n'
    f.write(string)
f.close()

descriptions = []
f = open('descriptions.txt', 'r')
for i in f:
    descriptions.append(i)

import pickle
file = open("features.pkl",'rb')
features = pickle.load(file)
file.close()
description ={}
for i in range(len(captions)):
    description[image[i]] = descriptions[i]
 

def max_length(descriptions):
    n = 0
    for i in descriptions:
        lst = i.split(" ")
        l = len(lst)
        if l>n:
            n = l
    return n


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical , plot_model
from keras.models import Model , Input ,Dense , LSTM ,Embedding ,Dropout
from keras.layers.merge import add


tokenize = Tokenizer()
tokenize.fit_on_texts(descriptions)
from numpy import array

def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, photo)
			yield [[in_img, in_seq], out_word]

def create_sequences(tokenizer, max_length, desc, img):
	x1, x2, y = list(), list(), list()
	# encode the sequence
	seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# split into input and output pair
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		x1.append(img)
		x2.append(in_seq)
		y.append(out_seq)
	return array(x1), array(x2), array(y)

# determine the maximum sequence length
max_length = max_length(descriptions)
print('Description Length: %d' % max_length)
 
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

img_name=[]
for i in features.keys():
    img_name.append(i)
des_img = list(description.keys())
descriptions = {}
for i in img_name:
    if i in des_img:
        descriptions[i] = description[i]

vocab_size = len(tokenize.word_index) + 1
model = define_model(vocab_size, max_length)
# define checkpoint callback
epochs = 20
steps = len(descriptions)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

generator = data_generator(descriptions, features, tokenize, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
for i in range(epochs):
    # create the data generator
    generator = data_generator(descriptions, features, tokenize, max_length)
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # save model
    model.save('model_' + str(i) + '.h5')
