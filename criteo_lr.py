#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import criteo_input

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


FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    field_size = 39
    feature_size = 117581
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            #--------build weights--------
            FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())

            #--------build feature--------
            filename = FLAGS.data_dir
            features,labels = criteo_input.input_fn(filename) 
            feat_ids  = features['feat_ids']
            feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
            feat_vals = features['feat_vals']
            feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

            #--------build model--------
            feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)      # None * F * 1
            y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)
            y_bias = FM_B * tf.ones_like(y_w, dtype=tf.float32)     # None * 1
            y = y_bias + y_w
            pred = tf.sigmoid(y)
            
            #--------build loss--------
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

            #--------build optimizer-------
            global_step = tf.Variable(0)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=FLAGS.train_dir,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 30000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                _, lossval, step = sess.run([train_op, loss, global_step])

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    format_str = ('%s: step %d, loss = %.2f')
                    print(format_str % (datetime.now(), step, lossval))

        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
