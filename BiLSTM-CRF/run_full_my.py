#!/usr/bin/env python3
import os
import config
# Full training on your data/my using GPU
base_dir = os.path.dirname(os.path.abspath(__file__))
config.data_dir = os.path.join(base_dir, 'data', 'my') + '/'
config.files = ['admin']
config.train_dir = config.data_dir + 'admin_train.npz'
config.test_dir = config.data_dir + 'admin_test.npz'
config.exp_dir = os.path.join(base_dir, 'experiments', 'my_full') + '/'
# training params
config.epoch_num = 50
config.batch_size = 32
# use GPU 0 by default; change if needed
config.gpu = '0'

if not os.path.exists(config.exp_dir):
    os.makedirs(config.exp_dir, exist_ok=True)

from run import simple_run

simple_run()
print('Full run finished')
