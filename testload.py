import numpy as np
import wfdb


from ecg import preprocess

'''
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load.load_dataset()
'''


def load_dataset(input_shape):
    '''load and form test sets'''
    '''---------changeable para-------'''
    class_num = 3
    subject_count = 150     # number of beats acquired from each subject, e.g. N: 150+150+150
    anno_skip = 5       # stride in sampling the labels
    ratio_trte = 0.1    # ratio of #training sets/ #test sets
    '''---------unchangeable para-------'''

    Type = ['N', '/', 'R']
    Srange = input_shape[0]
    freq = 360      # frequency = 360 Hz

    signal_pool = np.zeros((1, Srange, 1))
    # sig = np.zeros((1, Srange, 1))
    anno_pool = np.zeros((1, class_num))

    # subject_list = ['104', '105', '212']
    subject_list = ['102', '103', '118']        # same distribution
    label_list = ['/', 'N', 'R']

    # Acquire target samples from subjects
    print('Forming datasets ...')
    for i in range(len(subject_list)):
        count = 0
        signal, fields = wfdb.rdsamp('data/mitdb/'+subject_list[i], channels=[0])
        annotation = wfdb.rdann('data/mitdb/'+subject_list[i], 'atr')
        anno_position = len(annotation.sample) - 5
        while count < subject_count:
            if annotation.symbol[anno_position] == label_list[i]:
                count += 1
                sfrom = annotation.sample[anno_position] - int(Srange / 2)
                sto = sfrom + Srange
                sig = signal[sfrom:sto].T
                sig = sig.reshape((1, Srange, 1))
                signal_pool = np.append(signal_pool, sig, axis=0)
                one_hot = np.zeros((1, class_num))
                one_hot[0][Type.index(label_list[i])] = 1
                anno_pool = np.append(anno_pool, one_hot, axis=0)
            anno_position -= anno_skip
        print('Scan completed: MIT-BIH ' + subject_list[i] + ' (' + label_list[i] +')')


    # Re-ordering
    signal_pool = np.delete(signal_pool, 0, axis=0)
    anno_pool = np.delete(anno_pool, 0, axis=0)
    per = np.random.permutation(signal_pool.shape[0])
    signal_pool = signal_pool[per, :]
    anno_pool = anno_pool[per, :]

    training_num = int(signal_pool.shape[0] * ratio_trte)

    X_test_orig = signal_pool[training_num:]
    Y_test = anno_pool[training_num:]

    # Pre-processing for X
    X_test = preprocess.X_process(X_test_orig)

    print()
    return X_test, Y_test, class_num


Input_Shape = (200, 1)
X_test, Y_test, classes = load_dataset(Input_Shape)
