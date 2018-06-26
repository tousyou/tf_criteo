import tensorflow as tf
import numpy as np
import os
import criteo_input
import deep_model

field_size = 39
def main(_):
    with tf.Session() as sess:
        global_step = tf.get_variable('global_step', [],
                      initializer=tf.constant_initializer(0), trainable=False)
        dfm = deep_model.DeepFM() 
        filename = './criteo.txt'
        features,labels = criteo_input.input_fn(filename) 
        feat_ids  = features['feat_ids']
        feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])
        
        y = dfm.inference(feat_ids,feat_vals)
        auc,up_op = dfm.auc(y,labels) 
        loss = dfm.loss(y,labels)
        train_op = dfm.train(loss,global_step)
        sess.run(tf.global_variables_initializer())    
        sess.run(tf.local_variables_initializer())
        for i in range(10):
            _,lossval,step,aucval,_ = sess.run([train_op, loss, global_step, auc, up_op])
            print('step = {0}, loss = {1}, auc = {2}'.format(step,lossval,aucval))
            
        
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
