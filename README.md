# Documentação do Modelo de Rede Neural para Reconhecimento de Dígitos

Este é um exemplo de implementação de um modelo de rede neural utilizando a biblioteca TensorFlow para realizar o reconhecimento de dígitos manuscritos do conjunto de dados MNIST.

## Importando as bibliotecas necessárias

```python
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
```

Nesta seção, as bibliotecas necessárias para a implementação do modelo são importadas, incluindo NumPy para operações numéricas, Matplotlib para visualização, e TensorFlow para construção e treinamento de redes neurais.

## Importação e Visualização dos Dados

```python
# Importando Dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plotar Imagem
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
```

## Importação e Visualização dos Dados

```python
# Importando Dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plotar Imagem
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
```

Os dados do MNIST, que consistem em imagens de dígitos manuscritos e seus rótulos correspondentes, são carregados nesta etapa. Uma imagem de exemplo é plotada para visualização.

## Pré-processamento dos Dados
    
```python
# Tamanho da imagem
# x_train[0]
x_train[0].shape

# Normalização dos dados
x_train = tf.keras.utils.normalize(x_train, axis=1)  # dados entre 0 e 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # dados entre 0 e 1
```

As imagens são normalizadas para garantir que os valores dos pixels estejam na faixa de 0 a 1, facilitando o treinamento da rede neural.

## Construção do Modelo

```python
# Criando Modelo
model = tf.keras.models.Sequential()  # um modelo básico feed-forward
model.add(tf.keras.layers.Flatten())  # transforma a entrada 28x28 em 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # uma camada totalmente conectada simples, 128 unidades, ativação relu
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # uma camada totalmente conectada simples, 128 unidades, ativação relu
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # camada de saída. 10 unidades para 10 classes. Softmax para distribuição de probabilidade
```

O modelo de rede neural é construído nesta seção usando a API Sequential do Keras. Ele consiste em uma camada de achatamento (Flatten) para transformar as imagens 28x28 em um vetor unidimensional, seguido por duas camadas totalmente conectadas (Dense) com ativação ReLU e uma camada de saída com ativação Softmax.

## Compilação do Modelo
    
```python
model.compile(optimizer='adam',  # otimizador padrão para iniciar
              loss='sparse_categorical_crossentropy',  # cálculo do "erro". A rede neural visa minimizar a perda.
              metrics=['accuracy'])  # métrica a ser acompanhada
```

O modelo é compilado com um otimizador 'adam', uma função de perda 'sparse_categorical_crossentropy' e a métrica de acurácia.

## Treinamento do Modelo

```python
# Treino
model.fit(x_train, y_train, epochs=3)  # treina o modelo
```

O modelo é treinado com os dados de treinamento (imagens e rótulos) por três épocas.

## Avaliação do Modelo

```python
# Avaliação
val_loss, val_acc = model.evaluate(x_test, y_test)  # avalia o modelo
print(val_loss)  # perda
print(val_acc)  # acurácia
```

O modelo é avaliado com os dados de teste para verificar sua performance, e a perda (loss) e a acurácia são impressas no console.