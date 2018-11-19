# -*- coding: utf-8 -*-
import os
import json
import time
import logging

def save_option(option):
    option_path = os.path.join(option.save_dir, option.exp_name, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def logger_setting(exp_name, save_dir, debug):
    logger = logging.getLogger(exp_name)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')

    log_out = os.path.join(save_dir, exp_name, 'train.log')
    file_handler = logging.FileHandler(log_out)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger

class Timer(object):
    def __init__(self, logger, max_step, last_step=0):
        self.logger = logger
        self.max_step = max_step
        self.step = last_step

        curr_time = time.time()
        self.start = curr_time
        self.last = curr_time

    def __call__(self):
        curr_time = time.time()
        self.step += 1

        duration = curr_time - self.last
        remaining = (self.max_step - self.step) * (curr_time - self.start) / self.step / 3600
        msg = 'TIMER, duration(s)|remaining(h), %f, %f' % (duration, remaining)

        self.last = curr_time