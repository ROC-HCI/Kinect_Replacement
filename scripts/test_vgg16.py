from keras.models import model_from_json
from transformation import transform, image_transform
# import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_model(json_structure, weights):
  model = model_from_json(json_structure)
  model.load_weights(weights)
  return model

def test_model(args, test_dl):
  json_structure = '%s/model_structure.json' % args.modeldir
  weights = '%s/model_weights.h5' % args.modeldir

  model = load_model(json_structure, weights)

  log_fn = '%s/test_log.txt' % result_dir
  logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
  logging.info(args)

  nb_batch = int(math.ceil(len(test_dl)/args.batchsize))
  sum_loss = 0
  for batch in range(nb_batch):
    test_batch = test_dl[batch*args.batchsize:(batch+1)*args.batchsize]
    test_images_batch, test_joints_batch = transform(args, test_batch)
    loss, acc = model.test_on_batch(test_images_batch, test_joints_batch)
    print('B{}: loss:{}'.format(batch+1, loss))
    logging.info('B{}: loss:{}'.format(batch+1, loss))
    sum_loss += loss

  avg_loss = sum_loss/nb_batch
  return avg_loss

# def predict_model(model, test_input):
#   test_image, 
# def plot_joints_limbs(image, joints):
#   joints = joints.reshape((7,2))

#   # im = plt.imread(image)
#   implot = plt.imshow(image)

#   for j in joints:
#     plt.scatter([j[0]], [j[1]], c='r')

#   plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batchsize', type=int)
  parser.add_argument('--datadir')
  parser.add_argument('--modeldir')
  parser.add_argument('--channel', type=int)
  parser.add_argument('--size', type=int)
  parser.add_argument('--min_dim', type=int)
  parser.add_argument('--cropping', type=int)
  parser.add_argument('--crop_pad_inf', type=float)
  parser.add_argument('--crop_pad_sup', type=float)
  parser.add_argument('--shift', type=int)
  parser.add_argument('--joints_num', type=int)
  args = parser.parse_args()

  test_fn = '%s/test_joints.csv' % args.datadir
  test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

  test_model(args, test_dl)
  
  # datum = train_dl.split(',')
  # image, joints = image_transform(args, datum)
  # image = image.transpose((1,2,0))
  # print(joints)
  # plot_joints_limbs(image, joints)

  # img_fn = 'data/FLIC-full/images/%s' % (datum[0])
  # joints = np.asarray([int(float(p)) for p in datum[1:]])
  # plot_joints_limbs(img_fn, joints)


