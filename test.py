# from model import SimpleCNN
# # from datagen import generate_mixed_adverserial_batch
# # from datagen import generate_adversarial_batch
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist
# import numpy as np
# import csv
# import os



# print("[INFO] loading MNIST dataset...")
# (trainx, trainy), (testx, testy) = mnist.load_data()
# trainx = trainx / 255.0
# testx = testx / 255.0
# # add a channel dimension to the images
# trainx = np.expand_dims(trainx, axis=-1)
# testx = np.expand_dims(testx, axis=-1)
# # one-hot encode our labels
# trainy = to_categorical(trainy,10)
# testy = to_categorical(testy, 10)

# # initialize our optimizer and model
# print("[INFO] compiling model...")
# opt = Adam(lr=1e-3)
# model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# model.compile(loss="categorical_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])


# from torchattacks import PGD, PGDL2, PGDRS
# import torch
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # atk = PGD(model, device= device, eps=8/255, alpha=2/225, steps=2, random_start=True)
# # atk = PGDL2(model, device= device, eps=8/255, alpha=2/225, steps=2, random_start=True)
# atk = PGDRS(model, device= device, eps=8/255, alpha=2/225, steps=2)

# print(atk)


# trainX =  torch.tensor(trainx)
# trainY = torch.tensor(trainy)
# testX =  torch.tensor(testx)
# testY =  torch.tensor(testy)

# adv_images = atk(trainX, trainY)

# print("Done")





####################################################################################################################






from model import SimpleCNN
from datagen import generate_mixed_adverserial_batch
from datagen import generate_adversarial_batch
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
# import tensorflow_datasets as tfds
from model_loader import model_fitcher

import numpy as np
import csv
import os
# from torchattacks import PGD, PGDL2, PGDRS
import torch

import argparse
# Create ArgumentParser object 
parser = argparse.ArgumentParser(description="")
parser.add_argument("--attack", type=str, default='PGD', help="PGD, FGSM, FFGSM, RFGSM, SINIFGSM, VNIFGSM" )
parser.add_argument("--dataset", type=str, default="kmnist", help="mnist, cifar10, cifar100, fashion_mnist, kmnist" )


# kmnist_dataset = tfds.load("kmnist")


args = parser.parse_args()
attack = args.attack
dataset = args.dataset

args = parser.parse_args()
attack = args.attack

print('attack', attack)
# print(attack.__name__)

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
            

if dataset=="mnist" or dataset == "fashion_mnist":
    depth_value = 1
    data_set = mnist
    width_1 = 28
    height_1 = 28
    nu_clases = 10
    print("[INFO] loading "+ dataset + " dataset...")
    (trainX, trainY), (testX, testY) = data_set.load_data()
    path_datset = "D:/test knowledge distillation/adversarial-attacks-pytorch/kmnist/"  
    trainX1 = np.load(path_datset+"kmnist-train-imgs.npz")
elif dataset =="cifar10":
    data_set = cifar10
    depth_value = 3
    width_1 = 32
    height_1 = 32
    nu_clases = 10
elif dataset =="cifar100":
    data_set = cifar100
    depth_value = 3
    width_1 = 32
    height_1 = 32
    nu_clases = 100
    print("[INFO] loading "+ dataset + " dataset...")
    (trainX, trainY), (testX, testY) = data_set.load_data()
if dataset =="kmnist":
    path_datset = "E:/copy2_adversarial-attacks-pytorch/kmnist/"       
    width_1 = 28
    height_1 = 28
    nu_clases = 10
    depth_value = 1
    print("[INFO] loading "+ dataset + " dataset...")
    trainX1 = np.load(path_datset+"kmnist-train-imgs.npz")
    trainY1 = np.load(path_datset+"kmnist-train-labels.npz")
    testX1 = np.load(path_datset+"kmnist-test-imgs.npz")
    testY1 = np.load(path_datset+"kmnist-test-labels.npz")
    (trainX, trainY), (testX, testY) = (trainX1['arr_0'] , trainY1['arr_0']) , (testX1['arr_0'] , testY1['arr_0']) 
else:
    raise ValueError("Invalid dataset. Use 'mnist', 'cifar10', or 'cifar100'.")

# load MNIST dataset and scale the pixel values to the range [0, 1]
trainX = trainX / 255.0
testX = testX / 255.0
if dataset == "mnist" or dataset =="kmnist":
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
trainY = to_categorical(trainY,nu_clases)
testY = to_categorical(testY, nu_clases)


################# TRAIN MODEL ON SIMPLE DATASET #######################################
# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=width_1, height=height_1, depth= depth_value, classes=nu_clases)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=1)
# # make predictions on the testing set for the model trained on non-adversarial images
print("[INFO] Evaluate training network...")
(loss_N, acc_N) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_N, acc_N))

filename = 'clean'
save_dir = 'E:/copy2_adversarial-attacks-pytorch/saved_weights_kmnist2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = os.path.join(save_dir, filename)
model.save(path)



# ################# ADVERSERIAL GENERATION FULL  AND EVALUATION #######################################
# # generate a set of adversarial from our test set (so we can evaluate our model performance *before* and *after* mixed adversarial training)
# print("[INFO] generating adversarial examples..\n")

# (advX, advY) = next(generate_adversarial_batch(attack, model, len(testX),  testX, testY, (width_1, height_1, depth_value), eps=0.1))
# # re-evaluate the model on the adversarial images
# (loss_advdata_N_model, acc_advdata_N_model) = model.evaluate(x=advX, y=advY, verbose=0)
# print("[INFO] adversarial testing images on normal trained model:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_advdata_N_model, acc_advdata_N_model))


################# MIXED ADVERSERIAL GENERATION  AND TRAINING #######################################
# lower the learning rate and re-compile the model (such that we can fine-tune it on the mixed batches of normal images and dynamically generated adversarial images)
print("[INFO] re-compiling model for training with mixed adv and normal data...")
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] creating mixed data generator...")
(mixedImages, mixedLabels) =next(generate_mixed_adverserial_batch(attack, model, len(trainX), trainX, trainY, (width_1, height_1, depth_value), eps=0.1, split=0.5))
# (val_mixedImages, val_mixedLabels) =next(generate_mixed_adverserial_batch(attack, model, len(testX), testX, testY, (width_1, height_1, depth_value), eps=0.1, split=0.5))
# print('mixedImages',mixedImages.shape )
# print('mixedLabels',mixedLabels.shape )
# print('val_mixedImages', val_mixedImages.shape)
# print('val_mixedLabels', val_mixedLabels.shape)
# fine-tune our CNN on the adversarial images
print("[INFO] fine-tuning/training network on dynamic mixed data...")
model.fit( mixedImages,mixedLabels, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=0)

################ Saving the weights of the model #####################################################
filename = attack
save_dir = 'E:/copy2_adversarial-attacks-pytorch/saved_weights_kmnist2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = os.path.join(save_dir, filename)
model.save(path)

# Load the SavedModel
saved_model_path = path 
model = tf.keras.models.load_model(saved_model_path)

################# EVALUATION OF TEST DATA ON MIXED ADVERSERIAL GENERATION TRAINED NETWORK #######################################
# now that our model is fine-tuned we should evaluate it on the test set (i.e., non-adversarial) again to see if performance has degraded
(loss_N_data_adv_model, acc_N_data_adv_model) = model.evaluate(x=testX, y=testY, verbose=0)
print("")
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_N_data_adv_model, acc_N_data_adv_model))
################# EVALUATION OF ADV DATA ON MIXED ADVERSERIAL GENERATION TRAINED NETWORK #######################################
(loss_adv_data_adv_model, acc_adv_data_adv_model) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss_adv_data_adv_model, acc_adv_data_adv_model))

filename = 'D:/test knowledge distillation/adversarial-attacks-pytorch/output.dat'
writing_output( attack, float(round(loss_N,6)), float(round(acc_N,6)), float(round(loss_advdata_N_model,6)), float(round(acc_advdata_N_model,6)),  \
                float(round(loss_N_data_adv_model,6)), float(round(acc_N_data_adv_model,6)) ,float(round(loss_adv_data_adv_model,6)),\
                    float(round(acc_adv_data_adv_model,6)), filename  )


