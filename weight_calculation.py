#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:24:53 2023

@author: hayatu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:11:19 2023

@author: hayatu
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf



def calculate_cosine_similarity(v, z):
    
    # normalized_v = tf.nn.l2_normalize(v, axis=0)
    # normalized_z = tf.nn.l2_normalize(z, axis=0)
    
    # similarity = tf.reduce_sum(tf.multiply(normalized_v, normalized_z))

    # #similarity = np.array(similarity)
    similarity = cosine_similarity(v, z)
    return similarity

# Calculate weights based on cosine similarity scores
def calculate_weights(similarity_score):
    # Map similarity scores to weights using a transformation
    weights = 1 + similarity_score  
    return weights

# Normalize weights to ensure they sum up to 1
def normalize_weights(t_w):
    normalized_weights = []
    total_weights = np.reduce_sum(t_w)
    for i in range(len(t_w)):
        normalized_weights.append(t_w[i]/total_weights)
    return normalized_weights