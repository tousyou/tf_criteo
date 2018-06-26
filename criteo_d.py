#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import criteo_input
import deep_model


FLAGS = tf.app.flags.FLAGS
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string('data_dir', '',
                           """Path to the criteo data directory.""")
tf.app.flags.DEFINE_string('train_dir', '',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_integer("feature_size", 117581, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 10, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256_128', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.8_0.8', "dropout rate")
tf.app.flags.DEFINE_integer("use_fm", 1, "use fm model")
tf.app.flags.DEFINE_integer("use_deep", 1, "use deep_fm model")


def main(_):
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('num_epochs ', FLAGS.num_epochs)
    print('batch_size ', FLAGS.batch_size)
    print('log_steps ', FLAGS.log_steps)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)

    print('data_dir ',FLAGS.data_dir)
    print('train_dir ',FLAGS.train_dir)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    field_size = FLAGS.field_size
    feature_size = FLAGS.feature_size
    embedding_size = FLAGS.embedding_size
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    log_steps = FLAGS.log_steps
    learning_rate = FLAGS.learning_rate
    l2_reg = FLAGS.l2_reg
    deep_layers = FLAGS.deep_layers
    dropout = FLAGS.dropout

    use_fm = True
    use_deep = True
    if FLAGS.use_fm == 0:
        use_fm = False
    if FLAGS.use_deep == 0:
        use_deep = False
    print('use_fm = {0}, use_deep = {1}'.format(use_fm,use_deep))
     
    layers = FLAGS.deep_layers
    layers = [int(x) for x in layers.split('_')]
    dropout = FLAGS.dropout
    dropout = [float(x) for x in dropout.split('_')]

    filename = FLAGS.data_dir

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            dfm = deep_model.DeepFM(field_size = field_size,
                                    feature_size = feature_size,
                                    embedding_size = embedding_size,
                                    deep_layers = layers,
                                    dropout = dropout,
                                    l2_reg = l2_reg,
                                    use_fm = use_fm,
                                    use_deep = use_deep)
            #--------build feature--------
            features,labels = criteo_input.input_fn(filename,batch_size=batch_size,num_epochs=num_epochs)
            feat_ids  = features['feat_ids']
            feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
            feat_vals = features['feat_vals']
            feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

            #--------build model--------
            y = dfm.inference(feat_ids,feat_vals)

            #--------build auc--------
            auc,up_op = dfm.auc(y,labels)

            #--------build loss--------
            loss = dfm.loss(y,labels)

            #--------build train-------
            global_step = tf.train.get_or_create_global_step()
            train_op = dfm.train(loss,global_step)
            init_op = tf.global_variables_initializer()

            # The StopAtStepHook handles stopping after running given steps.
            hooks=[tf.train.StopAtStepHook(last_step=1000000)]
            
            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.train_dir,
                                           hooks=hooks) as mon_sess:
                mon_sess.run([init_op])
                while not mon_sess.should_stop():
                    try:
                        _, lossval, step, aucval, _ = mon_sess.run([train_op,loss,global_step, auc, up_op])
                        if step % log_steps == 0:
                            print('step ={0}, loss = {1}, auc = {2}'.format(step, lossval,aucval))
                    except tf.errors.OutOfRangeError:
                        print('****************one epoch**************')
                        break


if __name__ == "__main__":
    print('sys.argv = {0}'.format(sys.argv))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

