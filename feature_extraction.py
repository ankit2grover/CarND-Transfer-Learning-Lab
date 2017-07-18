import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from sklearn.preprocessing import LabelBinarizer
import datetime
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50,'The number of epochs')
flags.DEFINE_integer('batch_size', 256, 'Batch size')

now = datetime.datetime.now


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    n_classes = len(np.unique(y_train))
    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    #shape = X_train.reshape(512, n_classes)
    #fcW = tf.Variable(tf.truncated_normal(shape = shape, mean=0, stddev = 1e-2))
    #fcb = tf.Variable(tf.zeros(n_classes))
    #logits = tf.matmul(X_train, fcW) + fcb

    ## Flatten the ouput
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)

    ## One hot encode the output
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_val = label_binarizer.fit_transform(y_val)
    print(y_train.shape)
    print (y_val.shape)
    ## Define the Model
    model = Sequential()
    model.add(Flatten(input_shape= X_train.shape[1:]))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # TODO: train your model here
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    t = now()
    model.fit(X_train, y_train, nb_epoch = FLAGS.epochs, batch_size = FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)
    print("Training Time {}" .format((now() - t)))
    metrics = model.evaluate(X_val, y_val)
    for metrics_index in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metrics_index]
        metric_value = metrics[metrics_index]
        print ("{} and value is {}".format(metric_name, metric_value))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
