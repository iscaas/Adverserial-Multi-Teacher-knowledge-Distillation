#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:55:17 2023

@author: hayatu
"""

from model import SimpleCNN
import tensorflow as tf
from tensorflow import keras
import os


def model_fitcher(save_dir, models):
    pre_trained_teachers = []
    for i in range(len(models)):
        path = os.path.join(save_dir, models[i])
        model = tf.keras.models.load_model(path)
        pre_trained_teachers.append(model)
        
    return pre_trained_teachers
        
        
