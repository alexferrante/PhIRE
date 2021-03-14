import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

HR_DATA_DIR = 'prepared_data/tmp/01'
LR_DATA_DIR = 'prepared_data/tmp/LR'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_TFRecords(filename, data, mode='test', data_LR=None):
    '''
        Generate TFRecords files for model training or testing
        inputs:
            filename - filename for TFRecord (should by type *.tfrecord)
            data     - numpy array of size (N, h, w, c) containing data to be written to TFRecord
            model    - if 'train', then data contains HR data that is coarsened k times 
                       and both HR and LR data written to TFRecord
                       if 'test', then data contains LR data 
        outputs:
            No output, but .tfrecord file written to filename
    '''
    if mode == 'train':
        assert data_LR is not None

    with tf.io.TFRecordWriter(filename) as writer:
        for j in range(data.shape[0]):
            if mode == 'train':
                h_HR, w_HR, c = data[j, ...].shape
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tobytes()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data[j, ...].tobytes()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)})
            elif mode == 'test':
                h_LR, w_LR, c = data[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data[j, ...].tobytes()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                         'c': _int64_feature(c)})

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString()) 


hr_imgs = sorted(glob.glob(f'{HR_DATA_DIR}/*ct.png'))
HR_HT = 128
HR_WD = 128
C = 3
data_hr = np.ndarray(shape=(len(hr_imgs), HR_HT, HR_WD, C), dtype=np.int)
np_idx = 0
for hr_img in hr_imgs:
    img = Image.open(hr_img)
    img = np.asarray(img)
    img = img[:,:,:3]
    data_hr[np_idx] = img
    np_idx += 1

LR_HT = 64
LR_WD = 64
C = 3
lr_imgs = sorted(glob.glob(f'{LR_DATA_DIR}/*county.png'))
data_lr = np.ndarray(shape=(len(lr_imgs), LR_HT, LR_WD, C), dtype=np.int64)
np_idx = 0
for lr_img in lr_imgs:
    img = Image.open(lr_img)
    img = np.asarray(img)
    img = img[:,:,:3]
    data_lr[np_idx] = img
    np_idx += 1

print(data_hr.shape)
print(data_lr.shape)

generate_TFRecords('jan01_14data.tfrecord', data_hr, 'train', data_lr)