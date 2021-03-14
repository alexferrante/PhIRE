import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

HR_DATA_DIR = 'data_processing_util/prepared_data/tmp/01'
LR_DATA_DIR = 'data_processing_util/prepared_data/tmp/LR'

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def remove_transparency(filename, bg_colour=(255, 255, 255)):
    im = Image.open(filename)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        bg = bg.convert("RGB")
        bg.save(filename)
    else:
        return im

hr_imgs = sorted(glob.glob(f'{HR_DATA_DIR}/*ct.png'))
HR_HT = 128
HR_WD = 128
C = 3
data_hr = np.ndarray(shape=(len(hr_imgs), HR_HT, HR_WD, C))
np_idx = 0
for hr_img in hr_imgs:
    img_h = remove_transparency(hr_img)
    data_hr[np_idx] = img_h
    np_idx += 1

LR_HT = 64
LR_WD = 64
lr_imgs = sorted(glob.glob(f'{LR_DATA_DIR}/*county.png'))
data_lr = np.ndarray(shape=(len(lr_imgs), LR_HT, LR_WD, C))
np_idx = 0
for lr_img in lr_imgs:
    img_l = remove_transparency(lr_img)
    data_lr[np_idx] = img_l
    np_idx += 1

mode = 'train'
with tf.io.TFRecordWriter('jan01_14data.tfrecord') as writer:
  for j in range(data_hr.shape[0]):
    if mode == 'train':
      h_HR, w_HR, c = HR_HT,HR_WD, C
      h_LR, w_LR, c = LR_HT, LR_WD, C
      features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_lr[j,...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data_hr[j,...].tostring()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)})
    elif mode == 'test':
      h_LR, w_LR, c = LR_HT, LR_WD, C
      features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_lr[j,...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                         'c': _int64_feature(c)})
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString()) 