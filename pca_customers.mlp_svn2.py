# Pós em Tecnologias Disruptivas IESB
# Prof. Tatiana Saldanha Tavares 2018/2
# Banco de dado Clientes. Faz a PCA, atribui rótulos com K-means e em seguida classifica
# com Multi Layer Perceptron e Support Vector Machine

# Intalar bibliotecas
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import process_time

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the customers dataset
filename = 'customers.csv'
if not os.path.exists(filename):
    print('Arquivo não encontrado!!!')

try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print('customers dataset has {} samples with {} features each.'.format(*data.shape))
except:
    print('Dataset could not be loaded. Is the dataset missing?')

# normalização
#transforma uma lista em um vetor
X_ = np.array(data)
m = np.mean(X_)
s = np.std(X_,ddof=1)
X = (X_ - m)/s

#X = np.array(data) testar sem normaliza��o

#Calcular PCA de todo o conjunto de entrada

pca = PCA(n_components=5)
X_projected = pca.fit_transform(X)

print('Shape dados brutos: ' + str(X.shape))
print('Shape dados PCA: ' + str(X_projected.shape))

#Verificando o número de componentes
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('nÃºmero de componentes')
plt.ylabel('variÃ¢ncia cumulativa')
plt.show()

#Este banco de dados não é rotulado, preciso verificar os clusters e rotular.
# Set a KMeans clustering
kmeans = KMeans(n_clusters=6)
# Compute cluster centers and predict cluster indices
y = kmeans.fit_predict(X_projected)

# plot em 2D com 2 componentes
plt.scatter(X_projected[:, 0], X_projected[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Accent', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
plt.show()

# Separa conjunto de train e val dos dados brutos
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=42)

# Separa conjunto de train e val da PCA
projected_train, projected_val, y_train, y_val = train_test_split(X_projected, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=42)

# Definir arquitetura MLP
mlp = MLPClassifier(hidden_layer_sizes=(200,), activation='relu', max_iter=1000, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-3, random_state=1, learning_rate_init=.01)

# Aqui fazer um treinamento e uma validacao da MLP com dados brutos
start = process_time()
mlp.fit(X_train, y_train)
end = process_time()
time_mlp = end - start
print('Tempo de treinamento_mlp_dados_brutos: ' + str(time_mlp))
print('Erro no final do treinamento_mlp_dados_brutos: %f' % mlp.loss_)

# MÃ©tricas da validacÃ£o MLP com dados brutos
preds_val = mlp.predict(X_val)

correct_outputs_val = y_val
n_acertos_val = 0
for u in range(0, len(correct_outputs_val)):
   if preds_val[u] == correct_outputs_val[u]:
       n_acertos_val += 1
print('Number of acertos_val_mlp com dados brutos: ' + str((n_acertos_val*100)/len(correct_outputs_val)))
print(confusion_matrix(y_val,preds_val))
print(classification_report(y_val,preds_val))

# Aqui fazer um treinamento e uma validacao da MLP com PCA
start = process_time()
mlp.fit(projected_train, y_train)
end = process_time()
time_mlp = end - start
print('Tempo de treinamento_mlp_pca: ' + str(time_mlp))
print("Erro no final do treinamento_mlp_pca: %f" % mlp.loss_)

# MÃ©tricas da validacÃ£o MLP com PCA
preds_val = mlp.predict(projected_val)

correct_outputs_val = y_val
n_acertos_val = 0
for u in range(0, len(correct_outputs_val)):
   if preds_val[u] == correct_outputs_val[u]:
       n_acertos_val += 1
print('Number of acertos_val_mlp com PCA: ' + str((n_acertos_val*100)/len(correct_outputs_val)))
print(confusion_matrix(y_val,preds_val))
print(classification_report(y_val,preds_val))

## definir arquitetura da SVM
svm = svm.SVC(kernel='rbf',C =1, gamma='auto')

# Aqui fazer um treinamento e uma validacao da SVM com dados brutos
start = process_time()
svm.fit(X_train, y_train)
end = process_time()
time_svm = end - start
print('Tempo de treinamento_svm_dados_brutos: ' + str(time_svm))

# MÃ©tricas da validacÃ£o SVM com dados brutos
preds_val_svm= svm.predict(X_val)

correct_outputs_val_svm = y_val
n_acertos_val_svm = 0
for u in range(0, len(correct_outputs_val_svm)):
   if preds_val_svm[u] == correct_outputs_val_svm[u]:
            n_acertos_val_svm += 1
print('Number of acertos_val_svm_dados brutos: ' + str((n_acertos_val_svm*100)/len(correct_outputs_val_svm)))
print(confusion_matrix(y_val,preds_val_svm))

# Aqui fazer um treinamento e uma validacao da svm com PCA
start = process_time()
svm.fit(projected_train, y_train)
end = process_time()
time_svm = end - start
print('Tempo de treinamento_svm_pca: ' + str(time_svm))

# MÃ©tricas da validacÃ£o SVM com PCA
preds_val_svm= svm.predict(projected_val)

correct_outputs_val_svm = y_val
n_acertos_val_svm = 0
for u in range(0, len(correct_outputs_val_svm)):
   if preds_val_svm[u] == correct_outputs_val_svm[u]:
            n_acertos_val_svm += 1
print('Number of acertos_val_svm PCA: ' + str((n_acertos_val_svm*100)/len(correct_outputs_val_svm)))
print(confusion_matrix(y_val,preds_val_svm))

## testar um �nico exemplo
exemplo_=projected_val[3,]
exemplo = exemplo_.reshape((1, 5))
pred_svm = svm.predict(exemplo)
pred_mlp = mlp.predict(exemplo)