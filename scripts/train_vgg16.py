from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from sklearn.utils import shuffle
from vgg16 import vgg_16, vgg_16_conv, vgg_16_fc
from transformation import transform, image_transform

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
import h5py

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

def training(args, model, train_dl):
  for epoch in range(1, args.epoch + 1):
    #shuffle the training set before generating batches
    logging.info('Shuffling training set...')
    train_dl = np.random.permutation(train_dl)

    #devide training set into batches
    logging.info('Training epoch{}...'.format(epoch))
    nb_batch = int(math.ceil(len(train_dl)/args.batchsize))
    for batch in range(nb_batch):
      train_batch = train_dl[batch*args.batchsize:(batch+1)*args.batchsize]
      images_batch, joints_batch = transform(args, train_batch)
      print(images_batch.shape)
      print(joints_batch.shape)
      # images_batch = list(images_batch)
      # joints_batch = list(images_batch)
      loss = model.train_on_batch(images_batch, joints_batch)
      logging.info('batch{}, loss:{}'.format(batch+1, loss))
    
def save_bottleneck_features(args, train_dl):
  f = h5py.File(args.weights_path)
  conv_model = vgg_16_conv()

  for k in range(f.attrs['nb_layers']):
    if k >= len(conv_model.layers):
      break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    conv_model.layers[k].set_weights(weights)
  f.close()
  logging.info('Model loaded.')
  print('Model loaded')

  # nb_batch = int(math.ceil(len(train_dl)/args.batchsize))
  all_bottleneck_features = 0
  all_joints_info = 0
  # for batch in range(nb_batch):
  #   train_batch = train_dl[batch*args.batchsize:(batch+1)*args.batchsize]
  #   images_batch, joints_batch = transform(args, train_batch)
  #   print('Images batches and joints batches generated. Begin predicting...')
  #   print('Images batch shape:{}'.format(images_batch.shape))
  #   print('Joints batch shape:{}'.format(joints_batch.shape))
  #   batch_bottleneck_features = conv_model.predict_on_batch(images_batch)
  #   import pdb;pdb.set_trace()
  #   if batch == 0:
  #     all_bottleneck_features = batch_bottleneck_features
  #     all_joints_info = joints_batch
  #   np.append(all_bottleneck_features, batch_bottleneck_features)
  #   np.append(all_joints_info, joints_batch)

  for nb_dl, dl in enumerate(train_dl):
    image, joint = image_transform(args, dl.split(','))
    image = np.expand_dims(image, axis=0)
    bottleneck_features = conv_model.predict(image)
    if nb_dl == 0:
      all_bottleneck_features = bottleneck_features
      all_joints_info = joint
    else:
      all_bottleneck_features = np.concatenate((all_bottleneck_features, bottleneck_features), axis=0)
      all_joints_info = np.concatenate((all_joints_info, joint), axis=0)

  print('Saving bottleneck features...')
  logging.info('Saving bottleneck features...')
  np.save(open('all_bottleneck_features.npy', 'w'), all_bottleneck_features)
  np.save(open('all_joints_info.npy', 'w'), all_joints_info)

def train_fc_layers(args):
  train_data = np.load(open('all_bottleneck_features.npy'))
  train_joints = np.load(open('all_joints_info.npy'))
  print(train_data.shape)
  print(train_joints.shape)
  
  input_shape = train_data.shape[1:]
  fc_model = vgg_16_fc(input_shape, args.joints_num)
  fc_model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

  for epoch in range(1, args.epoch + 1):
    #shuffle the training set before generating batches
    logging.info('Shuffling training set...')
    train_data, train_joints = shuffle(train_data, train_joints)

    #devide training set into batches
    logging.info('Training epoch{}...'.format(epoch))
    nb_batch = int(math.ceil(len(train_data)/args.batchsize))
    for batch in range(nb_batch):
      data_batch = train_data[batch*args.batchsize:(batch+1)*args.batchsize]
      joints_batch = train_joints[batch*args.batchsize:(batch+1)*args.batchsize]
      loss = fc_model.train_on_batch(data_batch, joints_batch)
      logging.info('batch{}, loss:{}'.format(batch+1, loss))

def load_pretrain_weights(args, model):
  f = h5py.File(args.weights_path)

  for k in range(f.attrs['nb_layers']):
    if k >= 25:
      break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
  f.close()
  logging.info('Model loaded.')
  print('Model loaded')

  # set the first 25 layers to untrainable
  for layer in model.layers[0:25]:
    layer.trainable = False

  return model

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
  parser.add_argument('--joints_num', type=int)
  parser.add_argument('--symmetric_joints')
  parser.add_argument('--opt')
  parser.add_argument('--weights_path')
  args = parser.parse_args()

  # create result dir
  create_result_dir(args)

  # load datasets
  train_dl, test_dl = load_dataset(args)

  # load pre-trained model and train part of the model
  if args.weights_path:
    # get prediction from conv layers
    small_sample = train_dl[:1000]
    # save_bottleneck_features(args, small_sample)

    # train the dense layers
    logging.info('Training dense layers...')
    train_fc_layers(args)

    # model, opt = get_model_optimizer(args)
    # model = load_pretrain_weights(args, model)

    # model.compile(optimizer=opt,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    # training(args, model, train_dl)
  # otherwise train the entire model
  else:
    # create model and optimizer
    model, opt = get_model_optimizer(args)
    N, N_test = len(train_dl), len(test_dl)
    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    # compile the model
    logging.info('Compiling the model... Joints number: {}'.format(args.joint_num))
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    logging.info('Start training...')

    # training
    training(args, model, train_dl)
    logging.info('Training is done. Testing starts...')

  # testing
  # test_images, test_joints = transform(args, test_dl)
  # model.test_on_batch(test_images, test_joints)
  