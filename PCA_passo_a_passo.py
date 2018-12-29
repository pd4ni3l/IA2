#PCA do ponto de vista espacial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Cria dois vetores numpy.Cada vetor é um eixo dos dados de entrada. Temos N=10.
x = np.array([-2.5, -0.5, -2.2, -1.9, -3.1, -2.3, -2.0, -1.0, -1.5, -1.1])
y = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])

matrix_dados = np.array([x,y]).T
df = pd.DataFrame(matrix_dados, index=[i for i in range(1,len(x)+1)], columns=['x','y'])
print('Matriz de Dados - são 10 exemplos com 2 amostras cada')
print(df)

plt.title('Conjunto de Dados em 2 Dimensões')
plt.xlabel('Dimensão X')
plt.ylabel('Dimensão Y')
plt.xlim(-3.5, 0)
plt.ylim(0, 3.5)
plt.plot(x,y, 'o')
plt.show()

#O primeiro passo na execução do PCA consiste em normalizar os dados, isto é, subtrair 
#a média de cada uma das dimensões que caracterizam o conjunto de dados, de modo a obter um novo 
#conjunto, cuja a média é 0. Para este exemplo, é necessário calcular X-Média(X)  e  Y-Média(Y), 
#obtendo-se assim um novo conjunto de dados.

# Normalizar os dados - remover a média 
mediaX = np.mean(x)
mediaY = np.mean(y)
print('Média (X): ', mediaX)
print('Média (Y): ', mediaY)

x1 = x-np.mean(x)
y1 = y-np.mean(y)
matrix_dados_normalizados = np.array([x1,y1]).T
df = pd.DataFrame(matrix_dados_normalizados, index=[i for i in range(1,len(x1)+1)], columns=['x1','y1'])
print('Matriz de Dados Normalizados')
print(df)

plt.title('Dados Normalizados')
plt.xlabel('Dimensão X-Média(X)')
plt.ylabel('Dimensão Y-Média(Y)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot(x1,y1, 'o')
plt.show()

#Como é que todos os eixos do conjunto de dados se relacionam entre si?
#É necessário calcular as covariâncias entre todos os eixos do conjunto. 
#Para este exemplo em particular, por termos apenas um conjunto de dados composto por 
#2 eixos, a matriz de covariância será uma matriz 2x2 
convMatrix = np.cov(matrix_dados_normalizados.T)
print('Matriz de covariância')
df = pd.DataFrame(convMatrix, index=['x', 'y'], columns=['x','y'])
print(df)

#A cov(X,Y) < 0, como tal podemos deduzir que as dimensões X e Y 
#estão inversamente relacionadas entre si.

#Calcular os autovalores e autovetores da matriz de covariância
eigenvalues, eigenvectors = np.linalg.eig(convMatrix)
print('Autovalores: ',eigenvalues)
#o maior autovalor indica a posição da primeira componente principal
print('Autovetores')
df = pd.DataFrame(eigenvectors, index=['1', '2'], columns=['V1','V2'])
print(df)

#Os autovetores da matriz de covariância não dão informação acerca dos 
#padrões existentes nos dados. Ao projetarmos estes vetores sobre os dados 
#normalizados conseguimos visualizar mais facilmente estas informações.

#Note que os dados em si são os mesmos. Nos apenas rodamos para um novo set de 
#eixos (principais). Estamos literalmente olhado para os dados sob um novo ângulo. 
#Esse novo "ângulo" é mais intuitivo para tirarmos conclusões sobre os dados.

x = [i for i in range(-2,3)]
eigenV1 = [eigenvectors[1][0]*i/eigenvectors[0][0] for i in x]
eigenV2 = [eigenvectors[1][1]*i/eigenvectors[0][1] for i in x]

plt.arrow(0, 0, eigenvectors[0][0], eigenvectors[1][0], head_width=0.008, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, eigenvectors[0][1], eigenvectors[1][1], head_width=0.008, head_length=0.1, fc='k', ec='k')

plt.plot(x,eigenV1, linestyle='--', label='autovetor: V1')
plt.plot(x,eigenV2, linestyle='--', label='autovetor: V2')
plt.plot(x1,y1, 'o', label='Dados Normalizados')

plt.title('Projeção dos dados normalizados e seus autovetores')
plt.xlabel('Dimensão X-Média(X)')
plt.ylabel('Dimensão Y-Média(Y)')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
plt.show()

#Ao analisarmos o gráfico, reparamos que o autovetor V2 atravessa o conjunto dos dados,
# passando pelo meio dos pontos. Isto indica-nos o quão relacionado está este vetor com os dados 
# em análise. Desta análise, conseguimos inferir que o autovetor V2 é o componente principal, 
#caracterizando melhor os dados que o autovetor V1.
pca = PCA(n_components=1)
X_projected = pca.fit_transform(matrix_dados_normalizados)
plt.plot(X_projected, 'o')