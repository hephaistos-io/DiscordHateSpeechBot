#!python3
#coding:UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

import csv
import re
import sys
import pickle
import random
import zipfile

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
import numpy as np

def read_data(reprocess=False):
    if reprocess:
        print('Processing all the data')
        X, Y = [], []
        zip_ref = zipfile.ZipFile('train.csv.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()
        for i, row in enumerate(csv.reader(open('train.csv', encoding='UTF-8'))):
            if i > 0:   # Skip the header line
                sys.stderr.write('\r'+str(i))
                sys.stderr.flush()
                text = re.findall('\w+', row[1].lower())
                label = 1 if '1' in row[2:] else 0  # Any hate speach label 
                X.append(' '.join(text))
                Y.append(label)
        sys.stderr.write('\n')
        pickle.dump(X, open('X_tfsp.pkl', 'wb'))
        pickle.dump(Y, open('Y_tfsp.pkl', 'wb'))
    else:
        print('Loading preprocessed data')
        X = pickle.load(open('X_tfsp.pkl', 'rb'))
        Y = pickle.load(open('Y_tfsp.pkl', 'rb'))
    if debug:
        print(len(X), 'data points read')
        print('Label distribution:',Counter(Y))
        print('As percentages:')
        for label, count_ in Counter(Y).items():
            print(label, ':', round(100*(count_/len(X)), 2))
    return X, Y

def save_model(model, path):
    print('Saving model as ' + path)
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)

def create_figure(history, path):
    print('Creating history figure and saving with as ' + path)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.savefig(path, dpi=300)

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def validate_model(model, validation_data, labels):
    y_pred = model.predict(validation_data, batch_size=64, verbose=1)
    y_pred_bool = (y_pred >= 0)

    print(classification_report(labels,y_pred_bool, target_names=["non-hate speech", "hate-speech"]))
    print('Using Matthews correlation coefficient to measure quality')
    print('MCC-Quality: ', matthews_corrcoef(labels, y_pred_bool))

def turn_data_to_tensor(data, type):
    return tf.convert_to_tensor(data, dtype=type)

def combine_to_dataset(data, label):
    return tf.data.Dataset.from_tensor_slices((data, label))

def apply_shuffle_and_batch_to_dataset(dataset, shuffleSize, batchSize):
    return dataset.shuffle(shuffleSize).batch(batchSize).prefetch(tf.data.AUTOTUNE)

def create_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              #using tf.keras.metrics.Accuracy() here does not work as 
              # I T S N O T T H E S A M E E V E N T H O T H E N A M I N G S U G G E S T S I T
              metrics=['accuracy'])

    return model

def train_model(model, train_dataset, test_dataset, epoch):
    history = model.fit(train_dataset, epochs=epoch,
                    validation_data=test_dataset,
                    validation_steps=30)

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    return model, history

def printSystemInformation():
    print('System information:')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train tensorflow-network on hate-speech")
    parser.add_argument('-ds', '--downsample', help="downsample data, has to be a float between (0, 1) (exclusive)", type=float, default=0.8)
    parser.add_argument('-d', '--debug', help="turns debug on", action="store_true")
    parser.add_argument('-c', '--cpu', help="sets runmode to cpu only", action="store_true")
    parser.add_argument('-e', '--epochs', help="sets how many epochs the model shall train", type=int, default=10)
    parser.add_argument('-r', '--reprocess', help="activates the reprocessing of the training data. Has to be run the first time!", action="store_true")
    parser.add_argument('-rs', '--randomstate', help="sets a specific random state, used for the random.seed", type=int, default=42)
    parser.add_argument('-l', '--load', help="loads tensorflow model under the given path", default="")
    parser.add_argument('-ct', '--continuetraining', help="If a model to load has been passed, \
        this parameter will continue training it. IT is advised to supply a new -rs value, to mix up the training data", action="store_true")
    return parser.parse_args(argv)

def main(argv):

    print('HateSpeech Training with Tensorflow')

    global debug
    args = parse_args(argv)
    debug = args.debug
    random.seed(args.randomstate)
    if debug: 
        printSystemInformation()
        print('Setting np precision')
    np.set_printoptions(precision=4)

    print('Runing mode has been set to ' + 'CPU' if args.cpu else 'GPU')
    if debug: print('Setting up tensorflow')
    # setup spacy and set to preferr gpu
    if(not args.cpu):
        tf.device('GPU')
    else:
        tf.config.set_visible_devices([], 'GPU')

    print('Loading data...', file=sys.stderr)
    # read all data
    X_complete, Y_complete = read_data(reprocess=args.reprocess)
    print('Finished loading data')

    trainTestSplit = 0.2

    print('Downsampling data, only using ' + str(1-args.downsample))
    X, X_, Y, Y_ = train_test_split(X_complete, Y_complete, test_size=args.downsample, random_state=args.randomstate, stratify=Y_complete)

    print('Splitting data, to ' + str(1-trainTestSplit) + ' training data and using the remainder for testing')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=trainTestSplit, random_state=args.randomstate, stratify=Y)
    
    if debug: 
        print("train size: ", len(X_train))
        print("test size: ", len(X_test))

    
    print('Preparing verification data')
    X_test = turn_data_to_tensor(X_test, tf.string)
    Y_test = turn_data_to_tensor(Y_test, tf.int64)

    
    batchSize = 64

    if args.continuetraining or not args.load:
        print('Preparing training data')
        X_train = turn_data_to_tensor(X_train, tf.string)
        Y_train = turn_data_to_tensor(Y_train, tf.int64)
        train_dataset = combine_to_dataset(X_train, Y_train)
        test_dataset = combine_to_dataset(X_test, Y_test)
        train_dataset = apply_shuffle_and_batch_to_dataset(train_dataset, len(train_dataset), batchSize)
        test_dataset = apply_shuffle_and_batch_to_dataset(test_dataset, len(test_dataset), batchSize)

    if args.load:
        print('Loading existing model')
        model = tf.keras.models.load_model(args.load)
        if args.continuetraining:
            print('Retraining model')
            model, history = train_model(model, train_dataset, test_dataset, args.epochs) 
            if args.epochs > 1:
                create_figure(history, "models/graphs/tf_model" + str(datetime.now()) + "_retrained.png")
            savename = 'models/tf_model' + str(datetime.now()) + '_retrained'
            print('Saving retrained model as ' + savename)
            save_model(model, savename)
    else:
        print('Building new model')
        print('Preparing encoder data')
        X_complete = turn_data_to_tensor(X_complete, tf.string)
        Y_complete = turn_data_to_tensor(Y_complete, tf.int64)
        encoder_dataset = combine_to_dataset(X_complete, Y_complete)
    
        if debug:
            print('Random tests and labels:')
            for example, label in train_dataset.take(1):
                print('texts: ', example.numpy()[:3])
                print()
                print('labels: ', label.numpy()[:3])
                
        maxFeatures = 1000

        print('Building encoder with max_tokens:', maxFeatures)
        encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=maxFeatures)
        encoder.adapt(encoder_dataset.map(lambda text, label: text))

        print('Building model')
        model = create_model(encoder)
        
        print('Training model')
        model, history = train_model(model, train_dataset, test_dataset, args.epochs)
        if args.epochs > 1:
            create_figure(history, "models/graphs/tf_model" + str(datetime.now()) + ".png")
        savename = 'models/tf_model' + str(datetime.now())
        save_model(model, savename)
    
    print('Validating model')
    validate_model(model, X_test, Y_test)


    if debug:
        text = "I hate this piece of shit author"
        print('Test-prediction of: ' + text)
        print('Prediction: ', model.predict([text]))


if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))