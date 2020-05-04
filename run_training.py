#JupyterLab
#Neural Network

import tensorflow as tf
from tensorflow import keras

# import plaidml.keras
# plaidml.keras.install_backend()

#Bibioteka do obsługi sieci neuronowych
import keras

#Załadowania bazy uczącej
import imageio
import numpy as np

import os

ImgCount = 100
ImgWidth = 100
ImgHeight = 100

BazaImg = np.empty((ImgCount,ImgHeight,ImgWidth,3))
BazaAns = np.empty((ImgCount)) # 0 - m, 1 - k
dirList = os.listdir("C:\\Users\\Tymczasowy\\Desktop\\GenderClassifier\\baza\\m")
i=0
for dir in dirList:
  print(dir)
  FileName = "C:\\Users\\Tymczasowy\\Desktop\\GenderClassifier\\baza\\m\\{}".format(dir)
  Img = imageio.imread(FileName)
  Img = (Img / 127.5) - 1
  BazaImg[i,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]
  BazaAns[i] = 0
  i=i+1
  if i>=ImgCount/2:
    break

dirList = os.listdir("C:\\Users\\Tymczasowy\\Desktop\\GenderClassifier\\baza\\k")
for dir in dirList:
  print(dir)
  FileName = "C:\\Users\\Tymczasowy\\Desktop\\GenderClassifier\\baza\\k\\{}".format(dir)
  Img = imageio.imread(FileName)
  Img = (Img / 127.5) - 1
  BazaImg[i,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]
  BazaAns[i] = 1
  i=i+1
  if i>=ImgCount:
    break

BazaImg = BazaImg[0:i,:,:,:]
BazaAns = BazaAns[0:i]
print(BazaImg.shape)

#Stworzenia modelu sieci

input  = keras.engine.input_layer.Input(shape=(ImgHeight,ImgWidth,3),name="wejscie")

FlattenLayer = keras.layers.Flatten()

path = FlattenLayer(input)

for i in range(0,6):
  LayerDense1 = keras.layers.Dense(8, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
  path = LayerDense1(path)

  LayerPReLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
  path = LayerPReLU1(path)

LayerDenseN = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True, kernel_initializer='glorot_uniform')
output = LayerDenseN(path)

#---------------------------------
# Creation of TensorFlow Model
#---------------------------------
genderModel = keras.Model(input, output, name='genderEstimatior')

genderModel.summary() # Display summary

#Włączenia procesu uczenia

rmsOptimizer = keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999) #, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

genderModel.compile(optimizer=rmsOptimizer,loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

genderModel.fit(BazaImg, BazaAns, epochs=40, batch_size=10, shuffle=True)

genderModel.save('C:\\Users\\Tymczasowy\\Desktop\\GenderClassifier\\siec.h5')
#Przetestować / użyć sieci