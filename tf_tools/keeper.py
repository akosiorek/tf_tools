import os
import shutil
import tensorflow as tf
from tf_tools.util import try_mkdir


class Saver(object):
    def __init__(self, sess, checkpoint_dir, model_conf=None, data_conf=None, saver=None):
        self._sess = sess
        self._checkpoint_dir = checkpoint_dir
        self._model_conf = model_conf
        self._data_conf = data_conf

        if saver is None:
            saver = tf.train.Saver()

        self._saver = saver

        self._prepare()

    def _prepare(self):
        try_mkdir(self._checkpoint_dir)

        for property in ('_model_conf', '_data_conf'):
            path = getattr(self, property)
            if path is not None:
                basename = os.path.basename(path)
                dest = os.path.join(self._checkpoint_dir, basename)
                if not os.path.exists(dest):
                    shutil.copy(path, dest)
                    
                setattr(self, property, dest)




    def save(self, global_step=None):
        self._saver.save(self._sess, self._checkpoint_dir, global_step=global_step)


