import itertools
import os
import sys
import time
from datetime import datetime

import neurocity as nct

from attention.model import util
from attention.model.eval_tools import log_values
from attention.model.moat import *
from attention.model.loader import load_model, load_data

HOME = os.environ['HOME']
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('model_conf', None, '')
flags.DEFINE_string('data_conf', None, '')

flags.DEFINE_string('checkpoint_dir', None,
                           """Directory where to read model checkpoints.""")

flags.DEFINE_integer('n_batches', None,
                            """Number of samples to evaluate on.""")


# Flags governing the frequency of the eval.
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

flags.DEFINE_boolean('eval_all', False,
                            """Whether to evaluate all existing checkpoints""")

flags.DEFINE_boolean('eval_raw', False,
                            """Evaluate raw model (without restoring from snapshot)""")

flags.DEFINE_string('partition', 'test',
                           """Name added to logging tags""")

EVAL_FILE = 'eval_{}.txt'.format(FLAGS.partition)


def eval_file(checkpoint_dir):
    return os.path.join(checkpoint_dir, EVAL_FILE)


def log_evaluated_checkpoint(checkpoint_dir, n_iter, losses):
    line = '{}: {}\n'.format(n_iter, losses)
    with open(eval_file(checkpoint_dir), 'a') as f:
        f.write(line)


def last_evaluated_checkpoint(checkpoint_dir):
    try:
        with open(eval_file(checkpoint_dir), 'r') as f:
            lines = f.readlines()
            if not lines:
                return

            last_line = lines[-1]
            n_itr = int(last_line.split(':')[0])
            return n_itr

    except IOError:
        return


def _eval_once(global_step, model_file, eval_func, saver, writer):
    """Runs Eval once.
    Args:
      saver: Saver.
      writer: Summary writer.
    """
    with util.get_session(monitored=False) as sess:
        sess.run(tf.global_variables_initializer())
        if model_file is None:
            print 'Evaluating raw model'
        else:
            saver.restore(sess, model_file)
            print 'Successfully loaded model from {} at step={}.' \
                .format(model_file, global_step)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            threads = nct.start_queue_runners(sess=sess, coord=coord)

            print '{}: starting evaluation.'.format(datetime.now())
            values, elapsed_time = eval_func(sess)
            loss = values['loss/true/{}'.format(FLAGS.partition)]
            iou = values['loss/iou/{}'.format(FLAGS.partition)]

            print 'loss = {loss}, iou = {iou} at step {step}. Took {time}s.' \
                .format(loss=loss, iou=iou, time=elapsed_time, step=global_step)

            log_values(writer, global_step, dict=values)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        else:
            log_evaluated_checkpoint(FLAGS.checkpoint_dir, global_step, values)
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def _eval(eval_func, saver, writer):
    checkpoint_dir = FLAGS.checkpoint_dir
    last_evaluated = last_evaluated_checkpoint(checkpoint_dir)

    steps, model_files = [], []
    if FLAGS.eval_all:

        dated_dirs = util.find_dated_dirs(checkpoint_dir)
        dated_dirs = [os.path.join(checkpoint_dir, dd) for dd in dated_dirs]
        checkpoints = dict()
        for dd in dated_dirs:
            checkpoints.update(util.find_model_files(dd))

        if last_evaluated is not None:
            checkpoints = {k: v for k, v in checkpoints.iteritems() if k > last_evaluated}

        checkpoints = sorted(checkpoints.items(), key=lambda x: x[0])
        steps = [x[0] for x in checkpoints]
        model_files = [x[1] for x in checkpoints]

    else:
        latest_checkpoint = util.find_latest_checkpoint(checkpoint_dir)

        if latest_checkpoint is None:
            print 'No checkpoint file found'
            return

        global_step, model_file = latest_checkpoint
        if global_step == last_evaluated:
            print 'Skipping already evaluated checkpoint at step={}'.format(global_step)
            return

        steps.append(global_step)
        model_files.append(model_file)

    for step, model_file in zip(steps, model_files):
        _eval_once(step, model_file, eval_func, saver, writer)


def compute(sess, exprs):
    time_elapsed = 0.
    loss_values = {k: 0. for k in exprs}

    skipped = 0
    print '{}:'.format(FLAGS.n_batches),
    for i in xrange(1, FLAGS.n_batches + 1):
        print i,
        # flush stdout; otherwise it sometimes doesn't print the above
        sys.stdout.flush()

        start_time = time.time()
        try:
            values = sess.run(exprs)
        except Exception as err:
            if FLAGS.debug:
                raise err
            skipped += 1
        else:
            time_elapsed += time.time() - start_time
            for k, v in values.iteritems():
                loss_values[k] += v

    print
    for k, v in loss_values.iteritems():
        loss_values[k] = v / (FLAGS.n_batches - skipped)

    return loss_values, time_elapsed


def main(args):
    data, n_batches = load_data(FLAGS.data_conf)
    factory = load_model(FLAGS.model_conf, with_loss=True)
    model, loss = factory(*data)

    loss = {'/'.join((k, FLAGS.partition)): v for k, v in loss.iteritems()}
    if FLAGS.n_batches is None:
        FLAGS.n_batches = n_batches

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.checkpoint_dir)

    with util.get_session(monitored=False) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            nct.test_mode(sess)
        except: pass

    def eval_func(sess):
        return compute(sess, loss)

    if FLAGS.eval_raw:
        _eval_once(0, None, eval_func, saver, writer)
    else:
        while True:
            _eval(eval_func, saver, writer)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()
