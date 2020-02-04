import time
import tensorflow as tf


class Barrier(object):
  def __init__(self, worker_num, barrier_num, sleep_time_ms=10):
    self._worker_num = worker_num
    self._barrier_num = barrier_num
    self._sleep_time_ms = sleep_time_ms
    self._counter_vars = []
    self._counter_add_ops = []
    self._counter_reset_ops = []
    ps_device = '/job:ps/task:0/cpu:0'
    with tf.device(ps_device):
      for i in range(self._barrier_num):
        for j in range(self._worker_num):
          counter_var = tf.get_variable(
            'counter-{}_{}'.format(i, j),
            (),
            tf.int32,
            initializer=tf.zeros_initializer
          )
          self._counter_vars.append(counter_var)
          self._counter_add_ops.append(counter_var.assign_add(1, use_locking=True))
          self._counter_reset_ops.append(counter_var.assign(0, use_locking=True))

  def barrier_reset(self, session, worker_index, barrier_index):
    index = barrier_index * self._worker_num + worker_index
    session.run(self._counter_reset_ops[index])

  def barrier(self, session, worker_index, barrier_index, epoch):
    for task_index in range(self._worker_num):
      if task_index == worker_index:
        session.run(self._counter_add_ops[barrier_index * self._worker_num + worker_index])
      index = barrier_index * self._worker_num + task_index
      count = session.run(self._counter_vars[index])
      retry_num = 0
      while count < epoch:
        time.sleep(self._sleep_time_ms)
        retry_num += 1
        count = session.run(self._counter_vars[index])
        if retry_num == 1:
          tf.logging.info("{} wait for {} to be completed".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), task_index))