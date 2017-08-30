import os
import re
import random
import datetime

import tensorflow as tf
import numpy as np

import neurocity as nct


def as_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def try_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_logdir(checkpoint_dir, run_name=None):
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")

    if run_name is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, run_name)

    log_dir = os.path.join(checkpoint_dir, now)
    try_mkdir(log_dir)
    return log_dir


def get_latest_logdir(checkpoint_dir, run_name=None):
    if run_name is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, run_name)

    latest_dir = None
    dirs_by_date = find_dated_dirs(checkpoint_dir)
    if dirs_by_date:
        latest_dir = os.path.join(checkpoint_dir, dirs_by_date[0])
    return latest_dir


def get_or_create_latest_logdir(checkpoint_dir, run_name=None):
    logdir = get_latest_logdir(checkpoint_dir, run_name)
    if logdir is None:
        logdir = make_logdir(checkpoint_dir, run_name)

    return logdir


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def extract_itr_from_modelfile(model_path):
    return int(model_path.split('-')[-1].split('.')[0])


def find_model_files(model_dir):
    pattern = re.compile(r'.ckpt-[0-9]+$')
    model_files = [f.replace('.index', '') for f in os.listdir(model_dir)]
    model_files = [f for f in model_files if pattern.search(f)]
    model_files = {extract_itr_from_modelfile(f): os.path.join(model_dir, f) for f in model_files}
    return model_files


def find_dated_dirs(path):
    dirs = os.listdir(path)
    dirs_as_date = []
    for d in dirs:
        try:
            date = datetime.datetime.strptime(d, '%Y_%m_%d_%H.%M')
            dirs_as_date.append((date, d))
        except:
            pass

    dirs_by_date = sorted(dirs_as_date, key=lambda x: x[0], reverse=True)
    dirs_by_date = [x[1] for x in dirs_by_date]
    return dirs_by_date


def find_latest_checkpoint(checkpoint_dir):
    dirs_sorted_by_date = find_dated_dirs(checkpoint_dir)
    if not dirs_sorted_by_date:
        return None

    model_files = []
    for d in dirs_sorted_by_date:
        model_dir = os.path.join(checkpoint_dir, d)
        model_files = find_model_files(model_dir)
        if model_files:
            break

    if not model_files:
        return None

    model_itr = max(model_files.keys())
    model_file = model_files[model_itr]
    return model_itr, model_file


def try_resume_from_dir(sess, saver, checkpoint_dir, run_name):
    run_folder = os.path.join(checkpoint_dir, run_name)
    if not os.path.exists(run_folder):
        return 0

    dirs_sorted_by_date = find_dated_dirs(run_folder)

    for d in dirs_sorted_by_date:
        model_dir = os.path.join(run_folder, d)
        model_files = find_model_files(model_dir)
        if model_files:
            break

    latest_checkpoint = find_latest_checkpoint(run_folder)
    if latest_checkpoint is not None:
        model_itr, model_file = latest_checkpoint

        print 'loading from', model_file
        saver.restore(sess, model_file)

        return model_itr
    else:
        print 'No modefile to resume from. Starting at iter = 0.'
        return 0


def get_session(checkpoint_dir=None, allow_growth=True, mem_fraction=1.0, with_XLA=False, start_nct_runners=False, scaffold_kwargs=None,
                save_checkpoint_secs=1200, save_summaries_steps=500, monitored=True, **kwargs):

    if 'config' not in kwargs:

        kwargs['config'] = config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = mem_fraction
        config.graph_options.optimizer_options.do_function_inlining = True

        if with_XLA:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    if monitored:
        if scaffold_kwargs is None:
            scaffold_kwargs = dict()

        if start_nct_runners:
            init_fn = lambda a, b: None
            if 'init_fn' in scaffold_kwargs:
                init_fn = scaffold_kwargs['init_fn']

            def init_fun_start_nct_runners(scaffold, sess):
                init_fn(scaffold, sess)
                coord = tf.train.Coordinator()
                nct.start_queue_runners(sess, coord)

            scaffold_kwargs['init_fn'] = init_fun_start_nct_runners

        kwargs['scaffold'] = tf.train.Scaffold(**scaffold_kwargs)
        kwargs['checkpoint_dir'] = checkpoint_dir
        kwargs['save_summaries_steps'] = save_summaries_steps
        kwargs['save_checkpoint_secs'] = save_checkpoint_secs
        Session = tf.train.MonitoredTrainingSession
    else:
        Session = tf.Session

    return Session(**kwargs)

