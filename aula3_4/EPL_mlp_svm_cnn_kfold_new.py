#!/usr/bin/env /home/runner/anaconda3/bin/python3.6

import scipy.io as sio
import numpy as np
import pickle as pck
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from time import process_time
import copy

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense

from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier

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

Nclass = 2
epocas = 5
k = 2

# definir arquitetura da ANN - MLP - com uma camada intermediária
mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=epocas, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-3, random_state=1, learning_rate_init=.01)

# aqui definir a cnn com keras
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=5, input_shape=(quantidade_exemplos, quantidade_amostras)))
model.add(MaxPooling1D(pool_size=5 ))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(Nclass, activation='softmax'))

#com o k folds definidos, separa dados de validação e treinamento - repetir k vezes (folds)

time_train_mlp=[]
acuracia_mlp = []
precisao_mlp = []

time_train_cnn=[]
acuracia_cnn = []
precisao_cnn = []

time_train_svm=[]
acuracia_svm = []
precisao_svm = []

# Validação cruzada com k folds
skf = StratifiedKFold(k)

for train_index, test_index in skf.split(sinal,label): 
    #print("Train:", train_index, "Validation:", test_index) 
    x_treinamento, x_validacao = sinal[train_index], sinal[test_index] 
    y_treinamento, y_validacao = label[train_index], label[test_index]

# Faz x_treinamento e x_validação adequados para MLP e SVM    
    n_exemplos_treinamento = x_treinamento.shape[0]
    x_treinamento_vetor = x_treinamento.reshape((n_exemplos_treinamento, -1))
    
    n_exemplos_validacao = x_validacao.shape[0]
    x_validacao_vetor = x_validacao.reshape((n_exemplos_validacao, -1))

## Aqui fazer um treinamento de x épocas e uma validacao da MLP
#    start = process_time()
#    mlp.fit(x_treinamento_vetor, y_treinamento)
#    end = process_time()
#    time_mlp = end - start
#    print('Métricas do treinamento da MLP')
#    print('Tempo de treinamento_mlp com ' + str (epocas) + ' épocas: ' + str(time_mlp))      
#    print("Erro no final do treinamento: %f" % mlp.loss_)
#      
## Métricas da validacão mlp
#    preds_val_mlp = mlp.predict(x_validacao_vetor)  
#    print ('Métricas de uma validação MLP')
#    print("Acertos do conjunto de validação: %f" % mlp.score(x_validacao_vetor, y_validacao))
#    cm_val_mlp = confusion_matrix(y_validacao, preds_val_mlp)
#    print('Matriz de Confusão')
#    print(cm_val_mlp)
#    TP = cm_val_mlp[0,0]
#    FP = cm_val_mlp[0,1]
#    FN = cm_val_mlp[1,0]
#    TN = cm_val_mlp[1,1]
#
#    acuracia_mlp_ = (TP+TN)*100/(len(y_validacao))
#    precisao_mlp_ = TP*100/(TP+FP)
#    print('acurácia_mlp_:  '+ str(acuracia_mlp_))
#    print('precisao_mlp_:  '+ str(precisao_mlp_))
#    print('###################################################################')
## Usar no calculo das médias da mlp
#    time_train_mlp.append(time_mlp)
#    acuracia_mlp.append(acuracia_mlp_)
#    precisao_mlp.append(precisao_mlp_)
#    
##   pck.dump(mlp, open("trained_mlp_fold_number_" + str(n) + ".pickle", "wb"))       
#################################################################################################
### Aqui fazer um treinamento de x épocas e uma validacao da cnn
##    net_cnn = copy.deepcopy(cnn)
##    net_cnn.fit(x_treinamento, y_treinamento)
### Tempo de treinamento cnn
##    history = np.array(net_cnn.train_history_)
##    time_cnn= 0
##    for j in range(0, len(history)):
##        aux = history[j]
##        times_cnn = aux['dur']
##        time_cnn += times_cnn
##    print('Tempo de treinamento_cnn com ' + str (epocas) + ' épocas: ' + str (time_cnn))
### Métricas da validacão cnn
##    preds_val_cnn= net_cnn.predict(x_validacao)
##    correct_outputs_val_cnn = y_validacao
##    n_acertos_val_cnn = 0
##    for u in range(0, len(correct_outputs_val_cnn)):
##        if preds_val_cnn[u] == correct_outputs_val_cnn[u]:
##           n_acertos_val_cnn += 1
##    print ('Métricas de uma validação CNN')
##    print('Number of acertos_val_cnn: ' + str(n_acertos_val_cnn))
##    cm_val_cnn = confusion_matrix(y_validacao, preds_val_cnn)
##    print('Matriz de Confusão')
##    print(cm_val_cnn)
##    TP = cm_val_cnn[0,0]
##    FP = cm_val_cnn[0,1]
##    FN = cm_val_cnn[1,0]
##    TN = cm_val_cnn[1,1]
##
##    acuracia_cnn_ = (TP+TN)*100/(len(y_validacao))
##    precisao_cnn_ = TP*100/(TP+FP)
##    print('acurácia_cnn_:  '+ str(acuracia_cnn_))
##    print('precisao_cnn_:  '+ str(precisao_cnn_))
##    print('###################################################################')
### Usar no calculo das médias da cnn
##    time_train_cnn.append(time_cnn)
##    acuracia_cnn.append(acuracia_cnn_)
##    precisao_cnn.append(precisao_cnn_)
################################################################################################
### Aqui fazer um treinamento de x épocas e uma validacao da svm   
##    start = process_time()
##    svm.fit(x_treinamento_vetor, y_treinamento)
##    end = process_time()
##    time_svm = end - start
##    print('Tempo de treinamento_svm com ' + str (epocas) + ' épocas: ' + str(time_svm))
### Métricas da validacão svm
##    preds_val_svm= svm.predict(x_validacao_vetor)
##    correct_outputs_val_svm = y_validacao
##    n_acertos_val_svm = 0
##    for u in range(0, len(correct_outputs_val_svm)):
##        if preds_val_svm[u] == correct_outputs_val_svm[u]:
##            n_acertos_val_svm += 1
##    print ('Métricas de uma validação SVM') 
##    print('Number of acertos_val_svm: ' + str(n_acertos_val_svm))  
##    cm_val_svm = confusion_matrix(y_validacao, preds_val_svm)
##    print('Matriz de Confusão')
##    print(cm_val_svm)
##    TP = cm_val_svm[0,0]
##    FP = cm_val_svm[0,1]
##    FN = cm_val_svm[1,0]
##    TN = cm_val_svm[1,1]
##    acuracia_svm_ = (TP+TN)*100/(len(y_validacao))
##    precisao_svm_ = TP*100/(TP+FP)
##    print('acurácia_svm_:  '+ str(acuracia_svm_))
##    print('precisao_svm_:  '+ str(precisao_svm_)) 
##    print('###################################################################')
### Usar no calculo das médias da svm  
##    time_train_svm.append(time_svm)
##    acuracia_svm.append(acuracia_svm_)
##    precisao_svm.append(precisao_svm_)
####################################################################################
##media_time_train_mlp = sum(time_train_mlp) / float(len(time_train_mlp))
##media_acuracia_mlp = sum(acuracia_mlp) / float(len(acuracia_mlp))
##media_precisao_mlp = sum(precisao_mlp) / float(len(precisao_mlp))
## 
##media_time_train_cnn = sum(time_train_cnn) / float(len(time_train_cnn))
##media_acuracia_cnn = sum(acuracia_cnn) / float(len(acuracia_cnn))
##media_precisao_cnn = sum(precisao_cnn) / float(len(precisao_cnn))
##
##media_time_train_svm = sum(time_train_svm) / float(len(time_train_svm))
##media_acuracia_svm = sum(acuracia_svm) / float(len(acuracia_svm))
##media_precisao_svm = sum(precisao_svm) / float(len(precisao_svm))
##
##print('Tempo médio de treinamento MLP com ' + str(k) + ' kfold ' + str (media_time_train_mlp))
##print('Tempo médio de treinamento CNN com ' + str(k) + ' kfold ' + str (media_time_train_cnn))
##print('Tempo médio de treinamento SVM com ' + str(k) + ' kfold ' + str (media_time_train_svm))
##
##print('Médias das Validações com ' + str(k) + ' folds')
##print('Acurácia_mlp: ' + str(media_acuracia_mlp))
##print('Precisão_mlp: ' + str(media_precisao_mlp))
##
##print('Acurácia_cnn: ' + str(media_acuracia_cnn))
##print('Precisão_cnn: ' + str(media_precisao_cnn))
##
##print('Acurácia_svm: ' + str(media_acuracia_svm))
##print('Precisão_svm: ' + str(media_precisao_svm))
##
