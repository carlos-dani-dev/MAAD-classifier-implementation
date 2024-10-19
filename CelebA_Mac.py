import os
import math
import cv2 as cv
import PIL as pil
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
from keras import Model, activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ReLU,LeakyReLU, Softmax, BatchNormalization, Input, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())


df_atributos = pd.read_csv('list_attr_celeba.csv')
atributos = df_atributos.columns[1:]


def load_embeddings(celebA_embeddings_path, celebA_embeddings_filenames_path):
    # carregamento dos arrays numpy de embedding, dos arrays numpy de label e dos arrays numpy de filenames 
    #celebA_embeddings_path = celebA_embeddings#"/home/viplab/Documentos/Carlos D/embeddings/CelebA_emb_summarized_teste.npz"
    celebA_embeddings_filenames_path = celebA_embeddings_filenames_path#"/home/viplab/Documentos/Carlos D/embeddings/CelebA_att_filnames_teste.npz"

    celebA_embeddings = np.load(celebA_embeddings_path)['arr_0']
    celebA_labels = np.load(celebA_embeddings_path)['arr_1']
    celebA_filenames = np.load(celebA_embeddings_filenames_path)['arr_0']
    #celebA_filenames = np.load(celebA_embeddings_path)['arr_2']

    print(len(celebA_embeddings), ' de shape ', celebA_embeddings.shape)
    print(len(celebA_labels), ' de shape ', celebA_labels.shape)
    print(len(celebA_filenames), ' de shape ', celebA_filenames.shape)

    return celebA_embeddings, celebA_labels, celebA_filenames


def manipulating_embeddings(celebA_embeddings, celebA_labels, celebA_filenames):
    # ajustando o shape dos embeddings para entrada no modelo:

    print('Shape anterior: ', celebA_embeddings.shape)
    celebA_embeddings = np.reshape(celebA_embeddings, (15000, 512,))
    print('Novo shape: ', celebA_embeddings.shape)

    for label in celebA_labels:
        for i in range(0, len(label)):
            if label[i] == -1: label[i] = 0

    trainX, trainy, train_filenames = celebA_embeddings[0:10500], celebA_labels[0:10500], celebA_filenames[0:10500]
    testX, testy, test_filenames = celebA_embeddings[10500:12750], celebA_labels[10500:12750], celebA_filenames[10500:12750]
    validX, validy, valid_filenames = celebA_embeddings[12750:15000], celebA_labels[12750:15000], celebA_filenames[12750:15000]

    original_label = [trainy, testy, validy]
    output_label = list
    branch_labels = list()
    for label in original_label:
        for i in range(40):
            original_label_transposed = original_label.T
            branch_labels.append(np.array(original_label_transposed[i]))
        output_label.append(branch_labels)

    return [(trainX, output_label[0], train_filenames), (testX, output_label[1], test_filenames), (validX, output_label[2], valid_filenames)]


def subnet(shared_layers_output, i):
    
    att_branch = Dense(512, name='dense_'+str(i)+'_1')(shared_layers_output)
    att_branch = ReLU()(att_branch)
    att_branch = BatchNormalization()(att_branch)
    att_branch = Dropout(0.5)(att_branch)

    att_branch = Dense(512, name='dense_'+str(i)+'_2')(att_branch)
    att_branch = ReLU()(att_branch)
    att_branch = BatchNormalization()(att_branch)
    att_branch = Dropout(0.5)(att_branch)

    branch_output = Dense(1, name=atributos[i], activation='sigmoid')(att_branch)

    return branch_output


def multitask_model_definition(attributes_qtt=40):

    #Input
    input_layer = Input(shape=(512,), name='input_layer')
    
    #Camada compartilhada (1 única)
    shared_x = Dense(512, name='shared_dense_layer')(input_layer)
    shared_x = ReLU()(shared_x)
    shared_x = BatchNormalization()(shared_x)
    shared_x = Dropout(0.5)(shared_x)

    branch_outputs = list()
    for i in range(attributes_qtt):
        branch_outputs.append(subnet(shared_x, i))

    model = Model(input_layer, branch_outputs, name='model')

    return model


def compiling_model(model, attributes_qtt=40):
    metrics_dict = {atributos[i]:['Accuracy'] for i in range(attributes_qtt)}
    loss_dict = {atributos[i]:'binary_crossentropy' for i in range(attributes_qtt)}

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,  # Número de passos antes de aplicar o decaimento
    decay_rate=decay_rate,   # Fator de decaimento ajustado
    staircase=True     # Decaimento em degraus
    )

    epochs = 100
    initial_learning_rate = 1e-03
    batch_size = 128
    decay_rate = initial_learning_rate/epochs
    decay_steps = 100000

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=opt,
        loss=loss_dict,
        loss_weights=[1/attributes_qtt]*attributes_qtt,
        metrics=metrics_dict
    )


def training_model(model, trainX, testX, trainy_input, testy_input, batch_size, epochs):
    history = model.fit(
    x=trainX,
    y=trainy_input,
    validation_data=(testX, testy_input),
    batch_size=batch_size,
    verbose=1,
    epochs=epochs
#   callbacks=[tensorboard_callback]
    )

    model_name = 'modelo_CelebA_'+str(epochs)+'_'+str(batch_size)+'.keras'
    model.save('models/'+model_name)

    return history