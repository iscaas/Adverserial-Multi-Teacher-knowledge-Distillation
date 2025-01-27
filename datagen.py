# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:13:54 2023

@author: hayatu
"""

import torch
# from fgsm import generate_image_adversary
# from rfgsm import generate_image_adversary
# from pgdrsl2 import generate_image_adversary
# from sinifgsm import generate_image_adversary
# from PGD import generate_image_adversary
# from apgdt import APGD

from torchattacks import PGD, PGDL2, PGDRS, FGSM, FFGSM, RFGSM, SINIFGSM, VNIFGSM

from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

# generate batch of adversarial image for model evaluation
def generate_adversarial_batch(attack, model, total, images, labels, dims,
	eps=0.01):
	
	if attack == "PGD":
		attack = PGD
	elif attack == "PGDL2":
		attack = PGDL2
	elif attack == "FGSM":
		attack = FGSM
	elif attack == "FFGSM":
		attack = FFGSM
	elif attack == "RFGSM":
		attack = RFGSM
	elif attack == "SINIFGSM":
		attack = SINIFGSM
	elif attack == "VNIFGSM":
		attack = VNIFGSM
	else:
		raise ValueError("Invalid attack type. Use 'PGD', 'PGDL2', or 'PGDRS'.")

	(h, w, c) = dims
		
	model1 = model
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Convert to a torch tensor and move it to the GPU device
	images = torch.tensor(images)
	labels = torch.tensor(labels)
	atk = attack(model, device= device, eps=eps, steps=10)
 
	while True:
		perturbImages = []
		perturbLabels = []
		idxs = np.random.choice(range(0, len(images)), size=total,replace=False)
		for i in idxs:
			# print(i)
			image = images[i]
			label = labels[i]
			image_reshaped = image.unsqueeze(0)
			label_reshaped = label.unsqueeze(0)
			adversary_torch = atk(image_reshaped, label_reshaped)
			adversary_np = adversary_torch.detach().cpu().numpy()
			adversary = tf.convert_to_tensor(adversary_np, dtype=tf.float32)
			label_np = label.detach().cpu().numpy()
			labels_s = tf.convert_to_tensor(label_np, dtype=tf.float32)
			adversary = tf.reshape(adversary, (h, w, c))
			perturbImages.append(adversary)
			perturbLabels.append(labels_s )
		yield (np.array(perturbImages), np.array(perturbLabels))
        

# generate mixed batch of clean and adverarial images for adversarial training
def generate_mixed_adverserial_batch(attack, model, total, images, labels,
	dims, eps=0.01, split=0.5):
	
	if attack == "PGD":
		attack = PGD
	elif attack == "PGDL2":
		attack = PGDL2
	elif attack == "PGDRS":
		attack = PGDRS
	elif attack == "FGSM":
		attack = FGSM
	elif attack == "FFGSM":
		attack = FFGSM
	elif attack == "RFGSM":
		attack = RFGSM
	elif attack == "SINIFGSM":
		attack = SINIFGSM
	elif attack == "VNIFGSM":
		attack = VNIFGSM
	else:
		raise ValueError("Invalid attack type. Use 'PGD', 'PGDL2', or 'PGDRS'.")
    
	(h, w, c) = dims
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	images = torch.tensor(images)
	labels = torch.tensor(labels)
	atk = attack(model, device= device, eps=eps, steps=10)
	# the number of adversarial images to generate
	totalNormal = int(total * split)
	totalAdv = int(total * (1 - split))
	while True:

		idxs = np.random.choice(range(0, len(images)),
			size=totalNormal, replace=False)
		mixedImages = images[idxs]
		mixedLabels = labels[idxs]

		idxs = np.random.choice(range(0, len(images)), size=totalAdv, replace=False)
		for i in idxs:
			image = images[i]
			label = labels[i]
			image_reshaped = image.unsqueeze(0)
			label_reshaped = label.unsqueeze(0)
			adversary_torch = atk(image_reshaped, label_reshaped)

			adversary_np = adversary_torch.detach().cpu().numpy()
			adversary = tf.convert_to_tensor(adversary_np, dtype=tf.float32)
			label_np = label.detach().cpu().numpy()
			labels_s = tf.convert_to_tensor(label_np, dtype=tf.float32)
			mixedImages = np.vstack([mixedImages, adversary])
			mixedLabels = np.vstack([mixedLabels, labels_s])
		# shuffle the images and labels together
		(mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)
		# yield the mixed images and labels to the calling function
		yield (mixedImages, mixedLabels)