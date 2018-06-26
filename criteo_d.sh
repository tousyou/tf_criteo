py_cmd="Python/bin/python3 criteo_lr.py \
field_size=39 \
feature_size=117581 \
embedding_size=32 \
num_epochs=1 \
batch_size=64 \
learning_rate=0.0001 \
l2_reg=0.0001 \
deep_layers=256_128 \
dropout=0.8_0.8 \
log_steps=100
"

TensorFlow_Submit \
--appName=lr_criteo   \
--archives=hdfs://ns3-backup/user/dongsheng4/Python.zip#Python \
--files=./criteo_input.py,./deep_model.py,./criteo_lr.py \
--worker_memory=4096 \
--ps_memory=4096 \
--num_ps=1  \
--num_worker=2  \
--worker_cores=4 \
--worker_gpu_cores=0 \
--data_dir=hdfs://ns3-backup/user/dongsheng4/criteo/criteo.txt \
--train_dir=hdfs://ns3-backup/user/dongsheng4/criteo/train/ \
--command=$py_cmd

