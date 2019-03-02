from ecg import load

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization
from keras.layers import MaxPooling1D, Dropout, Flatten, Conv1D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


Input_Shape = (200, 1)

'''======================load data sets============================'''
X_train, Y_train, X_test, Y_test, classes = load.load_dataset(Input_Shape)

# Normalize image vectors
# X_train = X_train_orig/1.
# X_test = X_test_orig/1.

# Reshape
# Y_train = Y_train_orig.T
# Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


'''======================Build the model============================'''


def SimpleCnnModel(input_shape):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    Returns:
    model -- a Model() instance in Keras

    Hyper-parameter:
    filter length = 31
    filter num
    padding length (one side) = 15
    """
    filter_len = 30
    filter_num = 10
    k = 2       # filter_num = filter_num * k ^ layer_num

    stride_num = 1
    drop_prob = 0.2

    convlayer_in_num = 2
    convlayer_out_num = 2

    # densenum = 20
    outputclass = 3     # three classes: normal, rbbb, paced

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)
    X = X_input

    # ((CONV -> BN -> RELU -> DROPOUT) * N -> Maxpooling) * M
    for i in range(convlayer_out_num):
        for j in range(convlayer_in_num):
            filter_num = filter_num * k ^ i
            X = Conv1D(filter_num, filter_len, strides=stride_num,
                       kernel_initializer='he_normal', padding='same')(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
            X = Dropout(drop_prob)(X)
        X = MaxPooling1D(2)(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    # X = Dense(densenum, activation='relu')(X)
    X = Dense(outputclass, activation='sigmoid')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X)
    ### END CODE HERE ###

    return model


'''======================Train and test the model============================'''
'''
1. Create the model by calling the function above
2. Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])
3. Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)
4. Test the model on test data by calling model.evaluate(x = ..., y = ...)
'''

simplemodel = SimpleCnnModel(input_shape=Input_Shape)
simplemodel.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

simplemodel.fit(x=X_train, y=Y_train, epochs=10, batch_size=32)

preds = simplemodel.evaluate(x=X_test, y=Y_test)
print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

simplemodel.save('SimpleModel_v2.h5')

'''-------------------------------------------------------------------------------'''
# prediction

# img_path = 'images/my_image.jpg'

# img = image.load_img(img_path, target_size=(64, 64))
# imshow(img)

# x = np.expand_dims(x, axis=0)
# x = image.img_to_array(img)
# x = preprocess_input(x)

# print(happyModel.predict(x))

'''-------------------------------------------------------------------------------'''
# useful functions in Keras

'''model.summary(): prints the details of your layers in a table with the sizes of its inputs/outputs'''
simplemodel.summary()


'''plot_model(): plots your graph in a nice layout. 
You can even save it as ".png" using SVG() if you'd like to share it on social media ;). 
It is saved in "File" then "Open..." in the upper bar of the notebook.'''

# plot_model(simplemodel, to_file='HappyModel.png')
# SVG(model_to_dot(simplemodel).create(prog='dot', format='svg'))



