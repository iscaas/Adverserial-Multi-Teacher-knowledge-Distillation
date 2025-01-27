# # -*- coding: utf-8 -*-
# """
# Created on Thu Mar  2 15:17:36 2023

# @author: hayatu
# """

# from model import SimpleCNN
# from datagen import generate_mixed_adverserial_batch
# from datagen import generate_adversarial_batch
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.datasets import cifar10
# from distiller import Distiller
# from tensorflow import keras
# from model_loader import model_fitcher
# import numpy as np
# import csv
# import os


# attack_on_data = 'FGSM'
# attack_models = ['FGSM', 'FFGSM', 'RFGSM', 'PGD']
# attack_models = ['FGSM']
# # load MNIST dataset and scale the pixel values to the range [0, 1]
# print("[INFO] loading MNIST dataset...")
# (trainX, trainY), (testX, testY) = mnist.load_data()
# trainX = trainX / 255.0
# testX = testX / 255.0
# # add a channel dimension to the images
# trainX = np.expand_dims(trainX, axis=-1)
# testX = np.expand_dims(testX, axis=-1)
# # one-hot encode our labels

# trainY = to_categorical(trainY,10)
# testY = to_categorical(testY, 10)

# # instantiating Student Model
# student_model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# student_model.summary()

# # print("[INFO] generating adversarial examples with FGSM and kmnist dataset...\n")
# # (advX, advY) = next(generate_adversarial_batch(attack_on_data, student_model, len(testX),  testX, testY, (28, 28, 1), eps=0.1))

# model_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_kmnist/'
# model_dir = 'D:/test knowledge distillation/adversarial-attacks-pytorch/saved_weights_kmnist/'
# pre_trained_teachers = model_fitcher(model_dir, attack_models)
# #path = os.path.join(model_dir, attack_models[0])
# #model = tf.keras.models.load_model(path)

# print("Knowledge Distillation starts...")

# # Initialize and compile distiller
# distiller = Distiller(student=student_model, teacher=pre_trained_teachers)
# distiller.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=[keras.metrics.CategoricalAccuracy()],
#     student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
#     distillation_loss_fn=keras.losses.KLDivergence(),
#     alpha=0.1,
#     temperature=20)

# distillation_history = distiller.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)


# (loss, acc) = distiller.evaluate(testX, testY, verbose=1)

# opt = Adam(lr=1e-3)
# student_model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=opt,
# 	metrics=["accuracy"])


# ################ Saving the weights of the model #####################################################
# filename = 'student'
# filename2 = 'distiller'
# save_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_distiller_kmnist/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# path = os.path.join(save_dir, filename)
# path2 = os.path.join(save_dir, filename2)
# student_model.save(path)
# #distiller.save(path2)
# ##############################################################################


# (loss, acc) = student_model.evaluate(x=testX, y=testY, verbose=0)
# print("[INFO] student_model normal testing images after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# ###############################################################################
# print("[INFO] generating adversarial examples with FGSM...\n")
# attack_on_data = 'FGSM'
# (advX1, advY1) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[0], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# # re-evaluate the model on the adversarial images
# (loss, acc) = student_model.evaluate(x=advX1, y=advY1, verbose=0)
# print("[INFO] student_model FGSM adversarial testing images after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# ####################################################################################
# ###############################################################################
# print("[INFO] generating adversarial examples with FFGSM...\n")
# attack_on_data = 'FFGSM'
# (advX2, advY2) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[1], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# # re-evaluate the model on the adversarial images
# (loss, acc) = student_model.evaluate(x=advX2, y=advY2, verbose=0)
# print("[INFO] student_model FFGSM adversarial testing images after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# ####################################################################################
# ###############################################################################
# print("[INFO] generating adversarial examples with RFGSM...\n")
# attack_on_data = 'RFGSM'
# (advX3, advY3) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[2], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# # re-evaluate the model on the adversarial images
# (loss, acc) = student_model.evaluate(x=advX3, y=advY3, verbose=0)
# print("[INFO] student_model RFGSM adversarial testing images after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# ####################################################################################
# ###############################################################################
# print("[INFO] generating adversarial examples with PGD...\n")
# attack_on_data = 'PGD'
# (advX4, advY4) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[3], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# # re-evaluate the model on the adversarial images
# (loss, acc) = student_model.evaluate(x=advX4, y=advY4, verbose=0)
# print("[INFO] student_model PGD adversarial testing images after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# ####################################################################################








########################################################################################################################################################################

from model import SimpleCNN
from datagen import generate_mixed_adverserial_batch
from datagen import generate_adversarial_batch
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from distiller import Distiller
from distiller_mul import Distiller_mul
from tensorflow import keras
from model_loader import model_fitcher
import numpy as np
import csv
import os
attack_models = ['FGSM', 'FFGSM', 'RFGSM', 'PGD']
attack_models = ['FGSM']

dataset ="kmnist"
path_datset = "D:/test knowledge distillation/adversarial-attacks-pytorch/kmnist/" 
print("[INFO] loading "+ dataset + " dataset...")
trainX1 = np.load(path_datset+"kmnist-train-imgs.npz")
trainY1 = np.load(path_datset+"kmnist-train-labels.npz")
testX1 = np.load(path_datset+"kmnist-test-imgs.npz")
testY1 = np.load(path_datset+"kmnist-test-labels.npz")
(trainX, trainY), (testX, testY) = (trainX1['arr_0'] , trainY1['arr_0']) , (testX1['arr_0'] , testY1['arr_0']) 
# load MNIST dataset and scale the pixel values to the range [0, 1]
trainX = trainX / 255.0
testX = testX / 255.0
if dataset == "mnist" or dataset =="kmnist":
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
trainY = to_categorical(trainY,10)
testY = to_categorical(testY, 10)




model_dir = 'D:/test knowledge distillation/adversarial-attacks-pytorch/saved_weights_kmnist/'
attack_models = ['clean']
pre_trained_teacher_1 = model_fitcher(model_dir, attack_models)
pre_trained_teacher_clean = pre_trained_teacher_1[0] 
print("[INFO] generating adversarial examples with clean MODEL and PGD attack...\n")
attack_on_data = 'RFGSM'
(advX_clean_3, advY_clean_3) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_clean, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))






model_dir = 'D:/test knowledge distillation/adversarial-attacks-pytorch/saved_weights_kmnist/'
attack_models = ['FGSM']
pre_trained_teacher_1 = model_fitcher(model_dir, attack_models)
pre_trained_teacher_FGSM = pre_trained_teacher_1[0] 
print("[INFO] generating adversarial examples with FGSM MODEL and FGSM attack...\n")
attack_on_data = 'FGSM'
(advX_FGSM_1, advY_FGSM_1) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_FGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
print("[INFO] generating adversarial examples with FGSM MODEL and FFGSM attack...\n")
attack_on_data = 'FFGSM'
(advX_FGSM_2, advY_FGSM_2) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_FGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
print("[INFO] generating adversarial examples with FGSM MODEL and RFGSM attack...\n")
attack_on_data = 'RFGSM'
(advX_FGSM_3, advY_FGSM_3) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_FGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
print("[INFO] generating adversarial examples with FGSM MODEL and PGD attack...\n")
attack_on_data = 'PGD'
(advX_FGSM_4, advY_FGSM_4) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_FGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))



model_dir = 'D:/test knowledge distillation/adversarial-attacks-pytorch/saved_weights_kmnist/'

attack_models = ['FGSM', 'FFGSM', 'RFGSM', 'PGD']
pre_trained_teacher = model_fitcher(model_dir, attack_models)
# instantiating Student Model
student_model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# student_model.summary()
print("Knowledge Distillation starts multiple teachers...")
# Initialize and compile distiller
distiller = Distiller_mul(student=student_model, teacher=pre_trained_teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[keras.metrics.CategoricalAccuracy()],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=20)
distillation_history = distiller.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)
(loss, acc) = distiller.evaluate(testX, testY, verbose=1)
opt = Adam(lr=1e-3)
student_model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=opt,
	metrics=["accuracy"])
filename = 'student'
save_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_distiller_kmnist/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = os.path.join(save_dir, filename)
student_model.save(path)


attack_models = ['FGSM']
pre_trained_teacher_1 = model_fitcher(model_dir, attack_models)
pre_trained_teacher_FGSM = pre_trained_teacher_1[0] 
print("[INFO] generating adversarial examples with FGSM...\n")
attack_on_data = 'FGSM'
(advX1, advY1) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_FGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
# # instantiating Student Model
# student_model_FGSM = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# # student_model_FGSM.summary()
# print("Knowledge Distillation starts FGSM...")
# # Initialize and compile distiller
# distiller_FGSM = Distiller(student=student_model_FGSM, teacher=pre_trained_teacher_1)
# distiller_FGSM.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=[keras.metrics.CategoricalAccuracy()],
#     student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
#     distillation_loss_fn=keras.losses.KLDivergence(),
#     alpha=0.1,
#     temperature=20)
# distillation_history_FGSM = distiller_FGSM.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)
# (loss, acc) = distiller_FGSM.evaluate(testX, testY, verbose=1)
# opt = Adam(lr=1e-3)
# student_model_FGSM.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=opt,
# 	metrics=["accuracy"])
# filename = 'FGSM_student'
# save_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_distiller_kmnist/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# path = os.path.join(save_dir, filename)
# student_model_FGSM.save(path)



attack_models = ['FFGSM']
pre_trained_teacher_2 = model_fitcher(model_dir, attack_models)
pre_trained_teacher_FFGSM = pre_trained_teacher_2[0]
print("[INFO] generating adversarial examples with FFGSM...\n")
attack_on_data = 'FFGSM'
(advX2, advY2) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_FFGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
# # instantiating Student Model
# student_model_FFGSM = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# # student_model_FFGSM.summary()
# print("Knowledge Distillation starts FFGSM...")
# # Initialize and compile distiller
# distiller_FFGSM = Distiller(student=student_model_FFGSM, teacher=pre_trained_teacher_2)
# distiller_FFGSM.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=[keras.metrics.CategoricalAccuracy()],
#     student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
#     distillation_loss_fn=keras.losses.KLDivergence(),
#     alpha=0.1,
#     temperature=20)
# distillation_history_FFGSM = distiller_FFGSM.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)
# (loss, acc) = distiller_FFGSM.evaluate(testX, testY, verbose=1)
# opt = Adam(lr=1e-3)
# student_model_FFGSM.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=opt,
# 	metrics=["accuracy"])
# filename = 'FFGSM_student'
# save_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_distiller_kmnist/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# path = os.path.join(save_dir, filename)
# student_model_FFGSM.save(path)


attack_models = ['RFGSM']
pre_trained_teacher_3 = model_fitcher(model_dir, attack_models)
pre_trained_teacher_RFGSM = pre_trained_teacher_3[0]
print("[INFO] generating adversarial examples with RFGSM...\n")
attack_on_data = 'RFGSM'
(advX3, advY3) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_RFGSM, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
# # instantiating Student Model
# student_model_RFGSM = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# # student_model_RFGSM.summary()
# print("Knowledge Distillation starts RFGSM...")
# # Initialize and compile distiller
# distiller_RFGSM = Distiller(student=student_model_RFGSM, teacher=pre_trained_teacher_3)
# distiller_RFGSM.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=[keras.metrics.CategoricalAccuracy()],
#     student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
#     distillation_loss_fn=keras.losses.KLDivergence(),
#     alpha=0.1,
#     temperature=20)
# distillation_history_RFGSM = distiller_RFGSM.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)
# (loss, acc) = distiller_RFGSM.evaluate(testX, testY, verbose=1)
# opt = Adam(lr=1e-3)
# student_model_RFGSM.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=opt,
# 	metrics=["accuracy"])
# filename = 'RFGSM_student'
# save_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_distiller_kmnist/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# path = os.path.join(save_dir, filename)
# student_model_RFGSM.save(path)


attack_models = ['PGD']
pre_trained_teacher_4 = model_fitcher(model_dir, attack_models)
pre_trained_teacher_PGD = pre_trained_teacher_4[0]
print("[INFO] generating adversarial examples with PGD...\n")
attack_on_data = 'PGD'
(advX4, advY4) = next(generate_adversarial_batch(attack_on_data, pre_trained_teacher_PGD, len(testX),
    testX, testY, (28, 28, 1), eps=0.1))
# # instantiating Student Model
# student_model_PGD = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
# # student_model_PGD.summary()
# print("Knowledge Distillation starts PGD...")
# # Initialize and compile distiller
# distiller_PGD = Distiller(student=student_model_PGD, teacher=pre_trained_teacher_4)
# distiller_PGD.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=[keras.metrics.CategoricalAccuracy()],
#     student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
#     distillation_loss_fn=keras.losses.KLDivergence(),
#     alpha=0.1,
#     temperature=20)
# distillation_history_PGD = distiller_PGD.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)
# (loss, acc) = distiller_PGD.evaluate(testX, testY, verbose=1)
# opt = Adam(lr=1e-3)
# student_model_PGD.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=opt,
# 	metrics=["accuracy"])
# filename = 'PGD_student'
# save_dir = 'D:/copy_adversarial-attacks-pytorch/saved_weights_distiller_kmnist/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# path = os.path.join(save_dir, filename)
# student_model_PGD.save(path)







# print("###############################################################################")
# print("                     [INFO] CNN FGSM 1 T                     :")
# filename = 'FGSM_student'
# path = os.path.join(save_dir, filename)
# model = tf.keras.models.load_model(path)
# (loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
# print("[INFO] student_model normal testing images  clean after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = model.evaluate(x=advX1, y=advY1, verbose=0)
# print("[INFO] student_model normal testing images  FGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = model.evaluate(x=advX2, y=advY2, verbose=0)
# print("[INFO] student_model normal testing images FFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = model.evaluate(x=advX3, y=advY3, verbose=0)
# print("[INFO] student_model normal testing images RFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = model.evaluate(x=advX4, y=advY4, verbose=0)
# print("[INFO] student_model normal testing images PGD after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# print("###############################################################################")


# print("###############################################################################")
# print("                     [INFO] CNN FFGSM 1 T                     :")
# (loss, acc) = student_model_FFGSM.evaluate(x=testX, y=testY, verbose=0)
# print("[INFO] student_model normal testing images  clean after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_FFGSM.evaluate(x=advX1, y=advY1, verbose=0)
# print("[INFO] student_model normal testing images  FGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_FFGSM.evaluate(x=advX2, y=advY2, verbose=0)
# print("[INFO] student_model normal testing images FFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_FFGSM.evaluate(x=advX3, y=advY3, verbose=0)
# print("[INFO] student_model normal testing images RFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_FFGSM.evaluate(x=advX4, y=advY4, verbose=0)
# print("[INFO] student_model normal testing images PGD after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# print("###############################################################################")

# print("###############################################################################")
# print("                     [INFO] CNN RFGSM 1 T                     :")
# (loss, acc) = student_model_RFGSM.evaluate(x=testX, y=testY, verbose=0)
# print("[INFO] student_model normal testing images  clean after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_RFGSM.evaluate(x=advX1, y=advY1, verbose=0)
# print("[INFO] student_model normal testing images  FGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_RFGSM.evaluate(x=advX2, y=advY2, verbose=0)
# print("[INFO] student_model normal testing images FFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_RFGSM.evaluate(x=advX3, y=advY3, verbose=0)
# print("[INFO] student_model normal testing images RFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_RFGSM.evaluate(x=advX4, y=advY4, verbose=0)
# print("[INFO] student_model normal testing images PGD after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# print("###############################################################################")

# print("###############################################################################")
# print("                     [INFO] CNN PGD 1 T                     :")
# (loss, acc) = student_model_PGD.evaluate(x=testX, y=testY, verbose=0)
# print("[INFO] student_model normal testing images  clean after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_PGD.evaluate(x=advX1, y=advY1, verbose=0)
# print("[INFO] student_model normal testing images  FGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_PGD.evaluate(x=advX2, y=advY2, verbose=0)
# print("[INFO] student_model normal testing images FFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_PGD.evaluate(x=advX3, y=advY3, verbose=0)
# print("[INFO] student_model normal testing images RFGSM after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# (loss, acc) = student_model_PGD.evaluate(x=advX4, y=advY4, verbose=0)
# print("[INFO] student_model normal testing images PGD after training:")
# print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
# print("###############################################################################")

print("###############################################################################")
print("                     [INFO] CNN Multiteacher                   :")
(loss, acc) = student_model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] student_model normal testing images  clean after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
(loss, acc) = student_model.evaluate(x=advX1, y=advY1, verbose=0)
print("[INFO] student_model normal testing images  FGSM after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
(loss, acc) = student_model.evaluate(x=advX2, y=advY2, verbose=0)
print("[INFO] student_model normal testing images FFGSM after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
(loss, acc) = student_model.evaluate(x=advX3, y=advY3, verbose=0)
print("[INFO] student_model normal testing images RFGSM after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
(loss, acc) = student_model.evaluate(x=advX4, y=advY4, verbose=0)
print("[INFO] student_model normal testing images PGD after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
print("###############################################################################")




# print("[INFO] generating adversarial examples with FGSM...\n")
# attack_on_data = 'FGSM'
# (advX1, advY1) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[0], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# re-evaluate the model on the adversarial images
(loss, acc) = student_model.evaluate(x=advX1, y=advY1, verbose=0)
print("[INFO] student_model FGSM adversarial testing images after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
####################################################################################
###############################################################################
# print("[INFO] generating adversarial examples with FFGSM...\n")
# attack_on_data = 'FFGSM'
# (advX2, advY2) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[1], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# re-evaluate the model on the adversarial images
(loss, acc) = student_model.evaluate(x=advX2, y=advY2, verbose=0)
print("[INFO] student_model FFGSM adversarial testing images after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
####################################################################################
###############################################################################
# print("[INFO] generating adversarial examples with RFGSM...\n")
# attack_on_data = 'RFGSM'
# (advX3, advY3) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[2], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# re-evaluate the model on the adversarial images
(loss, acc) = student_model.evaluate(x=advX3, y=advY3, verbose=0)
print("[INFO] student_model RFGSM adversarial testing images after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
####################################################################################
###############################################################################
# print("[INFO] generating adversarial examples with PGD...\n")
# attack_on_data = 'PGD'
# (advX4, advY4) = next(generate_adversarial_batch(attack_on_data, pre_trained_teachers[3], len(testX),
#     testX, testY, (28, 28, 1), eps=0.1))
# re-evaluate the model on the adversarial images
(loss, acc) = student_model.evaluate(x=advX4, y=advY4, verbose=0)
print("[INFO] student_model PGD adversarial testing images after training:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))
####################################################################################