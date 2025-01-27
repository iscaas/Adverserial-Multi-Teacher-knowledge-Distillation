# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:17:36 2023

@author: hayatu
"""

from model import SimpleCNN
from datagen import generate_mixed_adverserial_batch
from datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import csv

import argparse
# Create ArgumentParser object 
parser = argparse.ArgumentParser(description="")
parser.add_argument("--attack", type=str,default="fgsm", help="fgsm, rfgsm,pgdrsl2, sinifgsm, PGD, SINIFGSM, VNIFGSM ")
args = parser.parse_args()

attack = args.attack

def writing_output( attack, loss_N, acc_N, loss_advdata_N_model, acc_advdata_N_model,   loss_N_data_adv_model, acc_N_data_adv_model ,loss_adv_data_adv_model, acc_adv_data_adv_model, completeName_successful  ):
        temp = ['attack : ', attack, '    ', '    ',
             'loss_N : ', loss_N, '    ', '    ', 'acc_N : ', acc_N, '    ', '    ',
             'loss_advdata_N_model: ', loss_advdata_N_model, '    ', '    ', 'acc_advdata_N_model: ', acc_advdata_N_model,'    ', '    ', 
             'loss_N_data_adv_model', loss_N_data_adv_model, '    ', '    ',
             'acc_N_data_adv_model: ', acc_N_data_adv_model, '    ', '    ', 'loss_adv_data_adv_model: ', loss_adv_data_adv_model, '    ', 
             'acc_adv_data_adv_model: ', acc_adv_data_adv_model]
        
        with open(completeName_successful, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(temp)
            csvfile.close()

# filename = 'D:/knowledge_distillation/output.dat'
# loss_N, acc_N, loss_advdata_N_model, acc_advdata_N_model,   loss_N_data_adv_model, acc_N_data_adv_model ,loss_adv_data_adv_model, acc_adv_data_adv_model = 0,0,0,0,0,0,0,0
# writing_output( attack, float(round(loss_N,6)), float(round(acc_N,6)), float(round(loss_advdata_N_model,6)), float(round(acc_advdata_N_model,6)),  \
#                 float(round(loss_N_data_adv_model,6)), float(round(acc_N_data_adv_model,6)) ,float(round(loss_adv_data_adv_model,6)),\
#                     float(round(acc_adv_data_adv_model,6)), filename  )




# load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# one-hot encode our labels
trainY = to_categorical(trainY,10)
testY = to_categorical(testY, 10)

# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=64,
	epochs=10,
	verbose=1)

# make predictions on the testing set for the model trained on
# non-adversarial images
(loss_N, acc_N) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_N, acc_N))
# generate a set of adversarial from our test set (so we can evaluate
# our model performance *before* and *after* mixed adversarial
# training)


print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(args, model, len(testX),
	testX, testY, (28, 28, 1), eps=0.1))
# re-evaluate the model on the adversarial images
(loss_advdata_N_model, acc_advdata_N_model) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_advdata_N_model, acc_advdata_N_model))

# lower the learning rate and re-compile the model (such that we can
# fine-tune it on the mixed batches of normal images and dynamically
# generated adversarial images)
print("[INFO] re-compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# initialize our data generator to create data batches containing
# a mix of both *normal* images and *adversarial* images
print("[INFO] creating mixed data generator...")
dataGen = generate_mixed_adverserial_batch(args, model, 64,
	trainX, trainY, (28, 28, 1), eps=0.1, split=0.5)
# fine-tune our CNN on the adversarial images
print("[INFO] fine-tuning network on dynamic mixed data...")
model.fit(
	dataGen,
	steps_per_epoch=len(trainX) // 64,
	epochs=10,
	verbose=1)

# now that our model is fine-tuned we should evaluate it on the test
# set (i.e., non-adversarial) again to see if performance has degraded
(loss_N_data_adv_model, acc_N_data_adv_model) = model.evaluate(x=testX, y=testY, verbose=0)
print("")
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_N_data_adv_model, acc_N_data_adv_model))
# do a final evaluation of the model on the adversarial images
(loss_adv_data_adv_model, acc_adv_data_adv_model) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss_adv_data_adv_model, acc_adv_data_adv_model))

filename = 'D:/test knowledge distillation/adversarial-attacks-pytorch/output.dat'
writing_output( attack, float(round(loss_N,6)), float(round(acc_N,6)), float(round(loss_advdata_N_model,6)), float(round(acc_advdata_N_model,6)),  \
                float(round(loss_N_data_adv_model,6)), float(round(acc_N_data_adv_model,6)) ,float(round(loss_adv_data_adv_model,6)),\
                    float(round(acc_adv_data_adv_model,6)), filename  )


