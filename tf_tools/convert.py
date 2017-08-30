import argparse
import numpy as np
import tensorflow as tf

from hart.model import util
from hart.model.attention_ops import FixedStdAttention
from hart.model.tracker import HierarchicalAttentiveRecurrentTracker as HART
from hart.model.nn import AlexNetModel, IsTrainingLayer

parser = argparse.ArgumentParser()

parser.add_argument('--alexnet_dir', default='../../../')
parser.add_argument('--checkpoint_path')
parser.add_argument('--old', default=False)

args = parser.parse_args()

batch_size = 1
img_size = 187, 621, 3
crop_size = 56, 56, 3

rnn_units = 100
norm = 'batch'
keep_prob = .75

img_size, crop_size = [np.asarray(i) for i in (img_size, crop_size)]
keys = ['img', 'bbox', 'presence']

bbox_shape = (1, 1, 4)

tf.reset_default_graph()
util.set_random_seed(0)

x = tf.placeholder(tf.float32, [None, batch_size] + list(img_size), name='image')
y0 = tf.placeholder(tf.float32, bbox_shape, name='bbox')
p0 = tf.ones(y0.get_shape()[:-1], dtype=tf.uint8, name='presence')

is_training = IsTrainingLayer()
builder = AlexNetModel(args.alexnet_dir, layer='conv3', n_out_feature_maps=5, upsample=False, normlayer=norm,
                       keep_prob=keep_prob, is_training=is_training)

model = HART(x, y0, p0, batch_size, crop_size, builder, rnn_units,
             bbox_gain=[-4.78, -1.8, -3., -1.8],
             zoneout_prob=(.05, .05),
             normalize_glimpse=True,
             attention_module=FixedStdAttention,
             debug=True,
             transform_init_features=True,
             transform_init_state=True,
             dfn_readout=True,
             feature_shape=(14, 14),
             is_training=is_training)


# convert model (from the one running on Shadow)
var_list = {}
if args.old:
    for v in tf.trainable_variables():
        name = v.name
        name = name.replace('HierarchicalAttentiveRecurrentTracker', 'MultiObjectAttentiveTracker')
        name = name.replace('IdentityLSTMCell', 'RNN/LSTMCell')
        name = name.split(':')[0]
        var_list[name] = v
else:
    for v in tf.trainable_variables():
        name = v.name.replace('HierarchicalAttentiveRecurrentTracker', 'MultiObjectAttentiveTracker')
        name = name.replace('IdentityLSTMCell', 'rnn/IdentityLSTMCell')
        name = name.split(':')[0]
        var_list[name] = v
saver = tf.train.Saver(var_list=var_list)


# saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver.restore(sess, args.checkpoint_path)


saver = tf.train.Saver()
saver.save(sess, args.checkpoint_path)
