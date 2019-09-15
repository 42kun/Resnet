# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  """从CIFAR10数据文件中读取和解析数据
  
  建议：如果您想要n路并行读取，请调用这个函数
  N次。这会给你N个独立的读取操作，这将提供更好的数据混合
  
  返回值：一个单例对象，拥有以下属性
    height:图片高
    width:图片宽
    depth:图片通道数
    key:描述文件名和记录号的标量字符串张量
    label:图片标签
    uint8image:uint8格式的图片数据
      
  """
  """生成对象"""
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  """获得数据基础参数，包括标签字节数，图片长宽与通道数"""
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  """创建FixedLengthRecordReader类，它能从文件中读出固定的长度"""
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  """filename_queue中所有文件中依次读取key，value
  （key为自动生成的顺序编号，为0，1，2。。。。
  value则为一条记录，包含lable与image）"""
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  """将原本为字符串类型的编码还原为tf.uint8型"""
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  """将record_bytes从0位置开始切label_bytes（1）位作为result.label"""
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  """将record_bytes从1位置开始切image_bytes位然后转变为result.depth*result.height*result.width三维张量"""
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  """将三维张量result.depth*result.height*result.width转变为result.height*result.width*result.depth"""
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  """返回result"""
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  """构造数据与标签队列
  
  参数:
    image:三维张量 [height, width, 3] type.float32
    label:一维张量 type.int32
    min_queue_examples:队列中最少数据数
    batch_size: 所需要的一批图片数
    
  返回值:
    images:图片 四维张量 [size，height, width, 3]
    labels:标签 一维张量
  """
  """16线程生成并获取数据，始终保持数据池中数据量大于min_queue_examples并小于min_queue_examples + 3 * batch_size"""
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  """为tensorboard提供图像"""
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  """随机裁剪图片"""
  distorted_image = tf.image.random_crop(reshaped_image, [height, width,3])

  # Randomly flip the image horizontally.
  """随机左右翻转图片（可能翻转，也可能不翻转）"""
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  """随机调整图片亮度"""
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  """随机调整图片对比度"""
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  """图片归一化"""
  float_image = tf.image.per_image_standardization(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  """用读操作为CIFAR生成数据以用于评估模型
  
  参数：
     eval_data：布尔型参数，表明我们生成的是训练数据（True）还是测试数据（False）
     data_dir: CIFAR数据目录（数据强制需要.bin结尾，否则会报错）
     batch_size: 需要生成的一批数据中的图片数量
     
  返回值：
     images:4D张量，【图片数量，图片长，图片宽（等于长），3（RGB通道）】
     labels: 1D张量，【图片数量】为图片的标签（0-9）
  """


  """检查是生成训练数据还是测试数据
  生成数据文件路径，加到filenames数组中
  规定1epoch的数量（1epoch等于使用训练集中的全部样本训练一次）"""
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  """检查是否有对应的数据文件，若没有则报出错误"""
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  """创建一个文件名队列并将filenames放在其中"""
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  """从文件名队列中的文件中读取一份数据"""
  read_input = read_cifar10(filename_queue)
  """获取图像数据并将其数据类型转换为tf.float32"""
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  """将图片裁剪为[height, width]大小（中心裁剪）"""
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  """归一化图像数据"""
  float_image = tf.image.per_image_standardization(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  """数据池中最小数据量，为0.4*epoch"""
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  """生成一批数据"""
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)
