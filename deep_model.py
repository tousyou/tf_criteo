"""
copy from
https://github.com/lambdaji/tf_repos/blob/master/deep_ctr/Model_pipeline/DeepFM.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DeepFM(object):
    def __init__(self,field_size=39,
                      feature_size=117581,
                      embedding_size=32,
                      deep_layers=[256,128],
                      dropout=[0.8,0.8],
                      l2_reg=0.0001):
        self.field_size = field_size
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
     
        #--------build weights--------
        self.FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        self.FM_W = tf.get_variable(name='fm_w', shape=[self.feature_size], initializer=tf.glorot_normal_initializer())
        self.FM_V = tf.get_variable(name='fm_v', shape=[self.feature_size, self.embedding_size], initializer=tf.glorot_normal_initializer())

    def inference(self,feat_ids,feat_vals):
        #--------build feature--------
        feat_ids = tf.reshape(feat_ids,shape=[-1,self.field_size])
        feat_vals = tf.reshape(feat_vals,shape=[-1,self.field_size])

        #--------build model--------
        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, feat_ids)              # None * F * 1
            y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

        with tf.variable_scope("Second-order"):
            embeddings = tf.nn.embedding_lookup(self.FM_V, feat_ids)             # None * F * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, self.field_size, 1])
            embeddings = tf.multiply(embeddings, feat_vals)                 #vij*xi
            sum_square = tf.square(tf.reduce_sum(embeddings,1))
            square_sum = tf.reduce_sum(tf.square(embeddings),1)
            y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)  # None * 1

        with tf.variable_scope("Deep-part"):
            deep_inputs = tf.reshape(embeddings,shape=[-1,self.field_size*self.embedding_size]) # None * (F*K)
            for i in range(len(self.deep_layers)):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=self.deep_layers[i], \
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), scope='mlp%d' % i)
                #if FLAGS.batch_norm:
                #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   
                #    #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=self.dropout[i])
                #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

            y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                         weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), scope='deep_out')
            y_d = tf.reshape(y_deep,shape=[-1])

        with tf.variable_scope("DeepFM-out"):
            y_bias = self.FM_B * tf.ones_like(y_d, dtype=tf.float32)     # None * 1
            y = y_bias + y_w + y_v + y_d
        return y

    def loss(self,y,labels):
        #--------build loss--------
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
                   self.l2_reg * tf.nn.l2_loss(self.FM_W) + \
                   self.l2_reg * tf.nn.l2_loss(self.FM_V)
        return loss

    def auc(self,y,labels):
        pred = tf.sigmoid(y)
        with tf.variable_scope('auc'):
            auc,op = tf.metrics.auc(labels=labels, predictions=pred)
        return auc,op

    def train(self,loss,global_step):
        #--------build optimizer-------
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

