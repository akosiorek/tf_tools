import os
import shutil
import tensorflow as tf

from tf_tools.util import try_mkdir
from tf_tools.loader import load_model, load_data


class Keeper(object):
    def __init__(self, checkpoint_dir, model_conf=None, data_conf=None, saver=None, saver_kwargs=None):
        self._checkpoint_dir = checkpoint_dir
        self._model_conf = model_conf
        self._data_conf = data_conf

        self._saver = saver
        self._saver_kwargs = dict() if saver_kwargs is None else saver_kwargs

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

    def save(self, sess, global_step=None):
        if self._saver is None:
            self._saver = tf.train.Saver(**self._saver_kwargs)

        self._saver.save(sess, self._checkpoint_dir, global_step=global_step)

    def load(self):
        data = load_data(self._data_conf)
        factory = load_model(self._model_conf)
        model = factory(data)
        return data, model


