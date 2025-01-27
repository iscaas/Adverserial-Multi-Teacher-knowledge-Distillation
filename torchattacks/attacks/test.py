from model import SimpleCNN
# from datagen import generate_mixed_adverserial_batch
# from datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import csv




print("[INFO] loading MNIST dataset...")
(trainx, trainy), (testx, testy) = mnist.load_data()
trainx = trainx / 255.0
testx = testx / 255.0
# add a channel dimension to the images
trainx = np.expand_dims(trainx, axis=-1)
testx = np.expand_dims(testx, axis=-1)
# one-hot encode our labels
trainy = to_categorical(trainy,10)
testy = to_categorical(testy, 10)

# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


from torchattacks import PGD, PGDL2, PGDRS
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# atk = PGD(model, device= device, eps=8/255, alpha=2/225, steps=2, random_start=True)
# atk = PGDL2(model, device= device, eps=8/255, alpha=2/225, steps=2, random_start=True)
atk = PGDRS(model, device= device, eps=8/255, alpha=2/225, steps=2)

print(atk)


trainX =  torch.tensor(trainx)
trainY = torch.tensor(trainy)
testX =  torch.tensor(testx)
testY =  torch.tensor(testy)

adv_images = atk(trainX, trainY)

print("Done")





####################################################################################################################

from model import SimpleCNN
from datagen import generate_mixed_adverserial_batch
from datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import csv

from torchattacks import PGD, PGDL2, PGDRS
import torch

import argparse
# Create ArgumentParser object 
parser = argparse.ArgumentParser(description="")
parser.add_argument("--attack", type=str,default="fgsm", help="fgsm, rfgsm,pgdrsl2, sinifgsm, PGD ")
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
            

# load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainY = to_categorical(trainY,10)
testY = to_categorical(testY, 10)


# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=1)
# make predictions on the testing set for the model trained on non-adversarial images
(loss_N, acc_N) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_N, acc_N))

################# ADVERSERIAL GENERATION FULL  AND EVALUATION #######################################
# generate a set of adversarial from our test set (so we can evaluate our model performance *before* and *after* mixed adversarial training)
print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(args, model, len(testX),  testX, testY, (28, 28, 1), eps=0.1))
# re-evaluate the model on the adversarial images
(loss_advdata_N_model, acc_advdata_N_model) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss_advdata_N_model, acc_advdata_N_model))


################# MIXED ADVERSERIAL GENERATION  AND TRAINING #######################################
# lower the learning rate and re-compile the model (such that we can fine-tune it on the mixed batches of normal images and dynamically generated adversarial images)
print("[INFO] re-compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] creating mixed data generator...")
dataGen = generate_mixed_adverserial_batch(args, model, 64, trainX, trainY, (28, 28, 1), eps=0.1, split=0.5)
# fine-tune our CNN on the adversarial images
print("[INFO] fine-tuning network on dynamic mixed data...")
model.fit( dataGen, steps_per_epoch=len(trainX) // 64, epochs=10, verbose=1)

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