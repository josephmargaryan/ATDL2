#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Keras Layer for weight quantization regularizers.

Karen Ullrich, Jan 2017
"""
import numpy as np

from keras import backend as K
from keras.layers import Layer

from helpers import special_flatten
from extended_keras import logsumexp

if K.backend() == "tensorflow":
    import tensorflow as tf

class GaussianMixturePrior(Layer):
    """A Gaussian Mixture prior for Neural Networks """
    def __init__(self, nb_components, network_weights, pretrained_weights, pi_zero, **kwargs):
        self.nb_components = nb_components
        self.network_weights = [K.flatten(w) for w in network_weights]
        self.pretrained_weights = special_flatten(pretrained_weights)
        self.pi_zero = pi_zero

        super(GaussianMixturePrior, self).__init__(**kwargs)

    def build(self, input_shape):
        J = self.nb_components

        # create trainable ...
        #    ... means
        init_mean = np.linspace(-0.6, 0.6, J - 1)
        self.means = self.add_weight(
            name='means',
            shape=(J - 1,),
            initializer=lambda shape, dtype=None: init_mean.astype('float32'),
            trainable=True
        )
        #   ... the variance (we will work in log-space for more stability)
        init_stds = np.tile(0.25, J)
        init_gamma = - np.log(np.power(init_stds, 2))
        self.gammas = self.add_weight(
            name='gammas',
            shape=(J,),
            initializer=lambda shape, dtype=None: init_gamma.astype('float32'),
            trainable=True
        )
        #   ... the mixing proportions
        init_mixing_proportions = np.ones((J - 1))
        init_mixing_proportions *= (1. - self.pi_zero) / (J - 1)
        self.rhos = self.add_weight(
            name='rhos',
            shape=(J - 1,),
            initializer=lambda shape, dtype=None: np.log(init_mixing_proportions).astype('float32'),
            trainable=True
        )
        # Note: add_weight() automatically registers these as trainable weights
        super(GaussianMixturePrior, self).build(input_shape)

    def call(self, x, mask=None):
        J = self.nb_components
        batch_size = K.shape(x)[0]

        loss = 0.
        # here we stack together the trainable and non-trainable params
        #     ... the mean vector
        means = K.concatenate([K.constant([0.]), self.means], axis=0)
        #     ... the variances
        precision = K.exp(self.gammas)
        #     ... the mixing proportions (we are using the log-sum-exp trick here)
        min_rho = K.min(self.rhos)
        mixing_proportions = K.exp(self.rhos - min_rho)
        mixing_proportions = (1 - self.pi_zero) * mixing_proportions / K.sum(mixing_proportions)
        mixing_proportions = K.concatenate([K.constant([self.pi_zero]), mixing_proportions], axis=0)

        # compute the loss given by the gaussian mixture
        for weights in self.network_weights:
            loss = loss + self.compute_loss(weights, mixing_proportions, means, precision)

        # GAMMA PRIOR ON PRECISION
        # ... for the zero component
        (alpha, beta) = (5e3, 20e-1)
        neglogprop = (1 - alpha) * K.gather(self.gammas, [0]) + beta * K.gather(precision, [0])
        loss = loss + K.sum(neglogprop)
        # ... and all other components
        alpha, beta = (2.5e2, 1e-1)
        idx = np.arange(1, J)
        neglogprop = (1 - alpha) * K.gather(self.gammas, idx) + beta * K.gather(precision, idx)
        loss = loss + K.sum(neglogprop)

        # Return loss with proper batch dimension shape (batch_size, 1)
        # Tile the scalar loss across the batch dimension
        return K.reshape(K.tile(K.expand_dims(loss, 0), [batch_size]), (batch_size, 1))

    def compute_loss(self, weights, mixing_proportions, means, precision):
        if K.backend() == "tensorflow":
            diff = tf.expand_dims(weights, 1) - tf.expand_dims(means, 0)
        else:
            diff = weights[:, None] - means  # shape: (nb_params, nb_components)
        unnormalized_log_likelihood = - (diff ** 2) / 2 * K.flatten(precision)
        Z = K.sqrt(precision / (2 * np.pi))
        log_likelihood = logsumexp(unnormalized_log_likelihood, w=K.flatten(mixing_proportions * Z), axis=1)

        # return the neg. log-likelihood for the prior
        return - K.sum(log_likelihood)

    def compute_output_shape(self, input_shape):
        """Return the output shape - a scalar loss value."""
        return (input_shape[0], 1)

    def get_output_shape_for(self, input_shape):
        """Legacy method for older Keras versions."""
        return self.compute_output_shape(input_shape)
