import tensorflow as tf
import numpy as np
import os
import criteo_input
import deep_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("use_fm", 1, "use fm model")
tf.app.flags.DEFINE_integer("use_deep", 1, "use deep_fm model")
field_size = 39
def main(_):
    with tf.Session() as sess:
        global_step = tf.get_variable('global_step', [],
                      initializer=tf.constant_initializer(0), trainable=False)
        use_fm = True
        use_deep = True
        if FLAGS.use_fm == 0:
            use_fm = False
        if FLAGS.use_deep == 0:
            use_deep = False
        print('use_fm = {0}, use_deep = {1}'.format(use_fm,use_deep))
        dfm = deep_model.DeepFM(use_fm=use_fm,use_deep=use_deep) 
        filename = './criteo.txt'
        features,labels = criteo_input.input_fn(filename,
                          batch_size=1000,
                          num_epochs=100) 
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
        count = 0
        while True:
            try:
                count += 1
                _,lossval,step,aucval,_ = sess.run([train_op, loss, global_step, auc, up_op])
                if count % 50 == 0:
                    print('step = {0}, loss = {1}, auc = {2}'.format(step,lossval,aucval))
            except tf.errors.OutOfRangeError:
                break
        
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
