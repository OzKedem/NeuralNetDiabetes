'''
This code has been written by Oz Kedem in 2022 for a computer science finals project
this file specifically is about the machine itself who detects
weather or not a person is going to be diabetic in the future
'''


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Loading All the Needed Libraries^^

def main ():

    '''
    Loading the Data as a table
    '''
    df = pd.read_csv('pima-indians-diabetes.csv', header=None)
    df.head()
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
         'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class']
    df.head()

    '''
    Loading the Data as a 2D array
    '''
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    dataset

    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    X_scaled = scale(X)
    print('Scaled_X:\n', X_scaled)

    # Split dataset into 'train' & 'test' sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    # (optional): one encoding??
    y_train = np_utils.to_categorical(y_train)
    print('Y_Train Encoded:\n', y_train)

    '''
    Defining the keras module
    '''
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='LeakyReLU'))
    model.add(Dense(8, activation='ReLU'))
    model.add(Dense(2, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=10)

    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy * 100))

    # make class predictions with the model
    predictions = np.argmax(model.predict(X_test), axis=-1)
    # summarize the first 5 cases
    for i in range(5):
        print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y[i]))

    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)

    accuracy_score(y_test, y_pred)

    print(model.metrics_names)

    print(history.history.keys())

    '''
    Graphing
    '''
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print (predictions)


main()