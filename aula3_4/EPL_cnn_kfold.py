#!/usr/bin/env /home/runner/anaconda3/bin/python3.6

import scipy.io as sio
import numpy as np
import pickle as pck
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from time import process_time
import copy
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout

#Input
sinal_ = sio.loadmat('sinal')
sinal_ = sinal_.get('sinal', 0)
#cria uma matriz
sinal = np.zeros(shape = (sinal_.shape[0], 1, sinal_.shape[1]))
#preenche a matriz com os valores desejados
for k in range(0, sinal_.shape[0]):
    sinal[k, 0] = sinal_[k]
quantidade_amostras = sinal.shape[2]
quantidade_exemplos = sinal.shape[0]

#Label
label_ = sio.loadmat('label')
label_ = label_.get('label', 0)
#cria uma matriz
label = np.zeros(shape = (label_.shape[0], ))
#preenche a matriz com os valores desejados
for k in range(0, label_.shape[0]):
    label[k] = label_[k, 0] - 1
label = label.astype(np.uint8)

print(sinal.shape)
print(label.shape)

# Separa treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(sinal,label, test_size=0.5)

X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[2])
y_train = y_train.reshape(1, y_train.shape[0])

X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[2])
y_val = y_val.reshape(1, y_val.shape[0])

#definir arquitetura DL - CNN - uma camadas convolucionais
cnn = Sequential()
cnn.add(Conv1D(filters=10, kernel_size=5, input_shape=(X_train.shape[1],X_train.shape[2])))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(30, activation='relu'))
cnn.add(Dense(y_train.shape[1], activation='softmax'))
cnn.compile(loss='mse', optimizer='adam', metrics=['mae'])


# Aqui fazer um treinamento de x épocas e uma validacao da cnn
net_cnn = copy.deepcopy(cnn)
net_cnn.fit(X_train, y_train, epochs=1000, batch_size=10)
score = net_cnn.evaluate(X_train, y_train, batch_size=10)

#Validação

preds_val_cnn = net_cnn.predict(X_val) 

# Métricas da validacão cnn

y_val = y_val.reshape(y_val.shape[1])
preds_val_cnn = preds_val_cnn.reshape(preds_val_cnn.shape[1])

correct_outputs_val_cnn = y_val
n_acertos_val_cnn = 0

for u in range(0, len(correct_outputs_val_cnn)):
    if preds_val_cnn[u] == correct_outputs_val_cnn[u]:
              n_acertos_val_cnn += 1
print ('Métricas de uma validação CNN')
print('Number of acertos_val_cnn: ' + str(n_acertos_val_cnn))
cm_val_cnn = confusion_matrix(y_val, preds_val_cnn)
print('Matriz de Confusão')
print(cm_val_cnn)
TP = cm_val_cnn[0,0]
FP = cm_val_cnn[0,1]
FN = cm_val_cnn[1,0]
TN = cm_val_cnn[1,1]

acuracia_cnn_ = (TP+TN)*100/(len(y_val))
precisao_cnn_ = TP*100/(TP+FP)
print('acurácia_cnn_:  '+ str(acuracia_cnn_))
print('precisao_cnn_:  '+ str(precisao_cnn_))
print('###################################################################')
