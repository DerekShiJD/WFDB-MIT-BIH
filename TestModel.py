import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt

from ecg import testload

Input_Shape = (200, 1)
X_test, Y_test, classes = testload.load_dataset(Input_Shape)

print("number of test examples = " + str(X_test.shape[0]))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

model = load_model('SimpleModel_v1.h5')

preds = model.evaluate(x=X_test, y=Y_test)

print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


# visualize the prediction process
Type = ['Sinus', 'Paced', 'Rbbb']

for i in range(20):
    x_pred = np.array(X_test[i])
    x_pred = x_pred.reshape(1, 200, 1)
    x_print = X_test[i]
    y_pred = Y_test[i]

    yr = model.predict(x_pred)

    max = yr[0][0]
    id = 0

    for j in range(2):
        if yr[0][j+1] > max:
            max = yr[0][j+1]
            id = j + 1

    for k in range(3):
        if y_pred[k] == 1:
            real = Type[k]
            break

    result = Type[id]

    #print('Beat type: ' + real + ' Prediction:  ' + result)
    plt.plot(x_print)
    plt.title('CNN-based Cardiac Arrhythmia Detection\nBeat type: ' + real + '\nPrediction:  ' + result)
    plt.show()
