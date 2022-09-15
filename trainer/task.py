#!/usr/bin/env python3

"""Train and evaluate the model

This file trains the model upon the training data and evaluates it with
the eval data.
It uses the arguments it got via the gcloud command.
"""

import os
import argparse
import logging

import tensorflow as tf

import trainer.data as data
import trainer.model as model

def scheduler(epoch, lr):
    """ The function reduces the learning rate exponentially
    after epoch is greater than 15.

    Parameters:
        epoch: epoch value
        lr: learning rate
    """
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_accuracy', this_max=True):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                print("saving best weights")
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()

def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the eval folder and trains your solution
    from the model.py file with it.

    Parameters:
        params: parameters for training the model
    """
    (train_data, train_labels) = data.create_data_with_labels("data/train/",train=True)
    (eval_data, eval_labels) = data.create_data_with_labels("data/eval/",train=False)

    img_shape = train_data.shape[1:]
    input_layer = tf.keras.Input(shape=img_shape, name='input_image')

    ml_model = model.solution(input_layer)

    if ml_model is None:
        print("No model found. You need to implement one in model.py")
    else:
        save_best_model = SaveBestModel()
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        ml_model.fit(train_data, train_labels,
                     batch_size=model.get_batch_size(),callbacks=[callback,save_best_model],
                     epochs=model.get_epochs(),  shuffle=True,
                     validation_data=(eval_data,eval_labels), verbose=1,workers=4)
        
        #set best weigts
        ml_model.set_weights(save_best_model.best_weights)
        ml_model.evaluate(eval_data, eval_labels, verbose=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level / 10)

    train_model(args)
