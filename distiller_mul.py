# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 19:51:14 2022

@author: hayatu
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import torch
from weight_calculation import calculate_cosine_similarity, calculate_weights, normalize_weights

class Distiller_mul(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller_mul, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller_mul, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data
        # Forward pass of multiple teachers
        teacher_predictions = []
        for i in range(len(self.teacher)):
            pred = self.teacher[i](x, training=False)
            teacher_predictions.append(pred)
        # teacher_predictions = sum(teacher_predictions[:]) / len(self.teacher)
        
        teachers_similarities = []
        for i in range(4):
            teachers_similarities.append(calculate_cosine_similarity(self.student(x, training=False), teacher_predictions[i]))
        
        
        teachers_weights = []
        for i in range(4):
            teachers_weights.append(calculate_weights(teachers_similarities[i]))
        
        print("Teacher weights : ",type(teachers_weights[0]))
        # Normalize weights
        normalized_teachers_weights = normalize_weights(teachers_weights)
        
        teachers_weighted_predictions = 0
        for i in range(4):
            teachers_weighted_predictions += teacher_predictions[i] * normalized_teachers_weights[i]
        
        
        # Forward pass of single teacher
        #teacher_predictions = self.teacher(x, training=False) # shape : (32, 10)
     

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teachers_weighted_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
