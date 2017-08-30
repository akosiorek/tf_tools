import os
import imp
import importlib
import tensorflow as tf
from copy import deepcopy


def _import_module(module_path_or_name):
    module, name = None, None
    if module_path_or_name.endswith('.py'):
        file_name = module_path_or_name
        module_path_or_name = os.path.basename(os.path.splitext(module_path_or_name)[0])
        module = imp.load_source(module_path_or_name, file_name)
    else:
        module = importlib.import_module(module_path_or_name)

    if module:
        name = module_path_or_name.split('.')[-1].split('/')[-1]

    return module, name


def _reparse_flags():
    # Ensure that tf.flags defined in the loaded model are parsed, but make sure that any values
    # set earlier are retained
    old_flags = deepcopy(tf.flags.FLAGS.__flags)
    tf.flags.FLAGS._parse_flags()
    tf.flags.FLAGS.__flags.update(old_flags)


def load_model(conf_path, name=None, **kwargs):
    """Load a tensorflow model from file and wrap it in a tf.Template.

    Loads a tensorflow model from file and wraps it into a tf.Template. The file
    has to define a `load_model` function, which builds the model. This function
    should take a list of tensors as an input; these tensors are the inputs to the
    resulting template

    :param conf_path: string, name or a filepath to a module
    :param name: string, name used to create the tf.Template. It's set to the name of the module if None
    :param kwargs: dict, any kwargs are passed on to the `load_model` function.
    :throw ValueError, if the module `module_name` does not define a `load_model` function.
    :return: tf.Template
    """

    module, _name = _import_module(conf_path)
    name = name if name is not None else _name

    try:
        builder_func = module.load_model
    except AttributeError:
        raise ValueError("A model config should specify 'load_model' function but no such function was "
                           "found in {}".format(module.__file__))

    _reparse_flags()
    print "Loading model '{}' from {}".format(module.__name__, module.__file__)
    return tf.make_template(name, builder_func, create_scope_now_=True, **kwargs)


def load_data(conf_path, *args, **kwargs):
    module, name = _import_module(conf_path)
    try:
        data_func = module.get_data
    except AttributeError:
        raise ValueError("A data config should specify 'get_data' function but no such function was "
                           "found in {}".format(module.__file__))

    _reparse_flags()
    return data_func(*args, **kwargs)

if __name__ == '__main__':
    builder = load_model('../model/hart.py')
    print builder