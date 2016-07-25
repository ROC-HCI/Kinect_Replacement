from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from vgg16 import vgg_16
from transformation import transform

import ctypes
import imp
import logging
import numpy as np
import os
import re
import shutil
import six
import sys
import time
import argparse
import math

def load_dataset(args):
  train_fn = '%s/train_joints.csv' % args.datadir
  test_fn = '%s/test_joints.csv' % args.datadir
  train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
  test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

  return train_dl, test_dl

def create_result_dir(args):
  result_dir = 'results/{}_{}'.format(
    'vgg16', time.strftime('%Y-%m-%d_%H-%M-%S'))
  if os.path.exists(result_dir):
    result_dir += '_{}'.format(np.random.randint(100))
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  log_fn = '%s/log.txt' % result_dir
  logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
  logging.info(args)

  args.log_fn = log_fn
  args.result_dir = result_dir

def get_model_optimizer(args):
  model = vgg_16(args.joint_num)

  if 'opt' in args:
    # prepare optimizer
    if args.opt == 'Adagrad':
      optimizer = optimizers.Adagrad(lr=args.lr)
    elif args.opt == 'SGD':
      optimizer = optimizers.SGD(lr=args.lr, momentum=0.9)
    elif args.opt == 'Adam':
      optimizer = optimizers.Adam(lr=args.lr)
    else:
      raise Exception('No optimizer is selected')

    # if args.resume_opt is not None:
    #   serializers.load_npz(args.resume_opt, optimizer)
    #   args.epoch_offset = int(
    #       re.search('epoch-([0-9]+)', args.resume_opt).groups()[0])

    return model, optimizer

  else:
    print('No optimizer generated.')
    return model

def training(args, model, train_images):
  for epoch in range(1, args.epoch + 1):
    #shuffle the training set before generating batches
    logging.info('Shuffling training set...')
    train_images = np.random.permutation(train_images)

    #devide training set into batches
    logging.info('Training epoch{}...'.format(epoch))
    for batch in range(nb_batch):
      train_batch = train_images[batch*args.batchsize:(batch+1)*args.batchsize]
      images_batch, joints_batch = transform(args, train_batch)
      loss = model.train_on_batch(images_batch, joints_batch)
      logging.info('batch{}, loss:{}'.format(batch+1, loss))
    
if __name__ == '__main__':
  sys.path.append('../../scripts')  # to resume from result dir
  sys.path.append('../../models')  # to resume from result dir
  sys.path.append('models')  # to resume from result dir

  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int)
  parser.add_argument('--batchsize', type=int)
  parser.add_argument('--lr', type=float)
  parser.add_argument('--snapshot', type=int)
  parser.add_argument('--datadir')
  parser.add_argument('--channel', type=int)
  parser.add_argument('--test_freq', type=int)
  parser.add_argument('--flip', type=int)
  parser.add_argument('--size', type=int)
  parser.add_argument('--min_dim', type=int)
  parser.add_argument('--cropping', type=int)
  parser.add_argument('--crop_pad_inf', type=float)
  parser.add_argument('--crop_pad_sup', type=float)
  parser.add_argument('--shift', type=int)
  parser.add_argument('--gcn', type=int)
  parser.add_argument('--joint_num', type=int)
  parser.add_argument('--symmetric_joints')
  parser.add_argument('--opt')
  args = parser.parse_args()

  # create result dir
  create_result_dir(args)

  # create model and optimizer
  model, opt = get_model_optimizer(args)
  train_dl, test_dl = load_dataset(args)
  N, N_test = len(train_dl), len(test_dl)
  logging.info('# of training data:{}'.format(N))
  logging.info('# of test data:{}'.format(N_test))

  #compile the model
  logging.info('Compiling the model... Joints number: {}'.format(args.joint_num))
  model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  #training
  logging.info('Start training...')
  training(args, model, train_dl)
  logging.info('Training is done. Testing starts...')

  #testing
  test_images, test_joints = transform(args, test_dl)
  model.test_on_batch(test_images, test_joints)
  