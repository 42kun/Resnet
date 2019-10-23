import tensorflow as tf
import cifar10_input
import numpy as np
import time

# 所有的形状均是在输入为[-1,3,32,32]的情况下产生

data_dir = "CIFAR/"  # 数据集地址
test_size = 10000  # 测试数量
batch_size = 1000  # 训练数量
Resnet_layer = 34  # 改网络结构在这里改，可以选择[18,34,50,101,152]
class_count = 10  # 类数量
training_step = 1000  # 训练次数
display_step = 25  # 展示间隔

# 设置超参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_decay', 0.0002, "l2权重")
tf.app.flags.DEFINE_float("bn_episode", 0.001, "bn相关，具体作用不是很理解")


# 生成可训练参数，正则化选项会在tf.GraphKeys.REGULARIZATION_LOSSES添加
def create_variable(shape, name=None):
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)  # l2正则化
    initial = tf.contrib.layers.xavier_initializer()  # 一种比较先进的初始化方法
    new_variables = tf.get_variable(name, shape=shape, initializer=initial, regularizer=regularizer)
    return new_variables


# bn层处理
def BN(data, is_training, name=None):
    """
    注意几点：
    训练时请用：
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    测试时请把is_training置为false
    保存时还需要一点其他黑科技，详情可见 https://www.cnblogs.com/hrlnw/p/7227447.html
    """
    return tf.layers.batch_normalization(data, name="%s_bn" % name, training=is_training)


# relu处理
def RL(data, name=None):
    return tf.nn.relu(data, name="%s_relu" % name)


# 先将数据进行bn处理，再进行relu处理
def BN_RL(data, is_training, name=None):
    bn = BN(data, is_training, name)
    bn_rl = RL(bn)
    return bn_rl


# 基础的卷积操作
def Conv(data, filter, kernel, stride=(1, 1), padding="SAME", name=None):
    """
    @data 输入的数据 tensor
    @filter 输入维度与输出维度，一个1*2列表
    @kernel 卷积核的形状，一个1*2列表
    @stride 滑动步长，一个1*2元组
    @pad 补零情况，字符串，可选"SAME"与"VALID"
    @name
    """
    shape = kernel + filter  # 卷积核形状+输入输出维度
    stride = [1, stride[0], stride[1], 1]
    fc_w = create_variable(shape, name="%s_weight" % name)
    fc_b = create_variable(filter[-1:], name="%s_bias" % name)
    return tf.nn.conv2d(data, fc_w, stride, padding=padding, name="%s_conv" % name) + fc_b


# 为卷积加上bn层
def Conv_BN(data, filter, kernel, stride=(1, 1), padding="SAME", is_training=True, name=None):
    cov = Conv(data, filter, kernel, stride, padding, name)
    cov_bn = BN(cov, is_training, name="%s_bn" % name)
    return cov_bn


# 为卷积加上bn层与RL层
def Conv_BN_RL(data, filter, kernel, stride=(1, 1), padding="SAME", is_training=True, name=None):
    cov_bn = Conv_BN(data, filter, kernel, stride, padding, is_training, name)
    cov_bn_rl = RL(cov_bn)
    return cov_bn_rl


# 先加上BN_RL再conv
def BN_RL_Conv(data, filter, kernel, stride=(1, 1), padding="SAME", is_training=True, name=None):
    bn_rl = BN_RL(data, is_training=False, name=name)
    bn_rl_conv = Conv(bn_rl, filter, kernel, stride, padding, name=name)
    return bn_rl_conv


# 池化层，包装了一下
def Pooling(data, pool_type="avg", kernel=(2, 2), padding="SAME", stride=(2, 2), name=None):
    """
    @data 输入的数据 tensor
    @pool_type 池化类型 avg或者max
    kernel 池化窗口大小 1*2元组
    padding
    stride 滑动步长 1*2元组
    """
    ksize = [1, kernel[0], kernel[1], 1]
    stride = [1, stride[0], stride[1], 1]
    if pool_type == "avg":
        return tf.nn.avg_pool2d(data, ksize, stride, padding, name="%s_pool" % name)
    elif pool_type == "max":
        return tf.nn.max_pool2d(data, ksize, stride, padding, name="%s_pool" % name)
    else:
        raise Exception


# 残差块构造
# 我们要构造两种残差块，下面为第一种
def residual_block(input_layer, output_channel, first_block=False, is_training=True, name=None):
    with tf.variable_scope(name):
        # 获取input_layer输入最后一维
        stride = None
        input_channel = input_layer.get_shape().as_list()[-1]
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1

        # 如果是第一块
        # 据说第一块不需要进行batch_normalize,谁知道呢，看了3份代码3份实现都不一样
        # 之前看的参数表貌似是依据ImageNet的，图像很大。我们训练小网络理应对参数进行一些微调。保持大致结构就够了
        with tf.variable_scope("first"):
            conv1 = None
            if first_block:
                conv1 = Conv(input_layer, [input_channel, output_channel], [3, 3], [stride, stride], padding="SAME",
                             name=name)
            else:
                conv1 = BN_RL_Conv(input_layer, [input_channel, output_channel], [3, 3], [stride, stride],
                                   padding="SAME",
                                   is_training=is_training, name=name)
        with tf.variable_scope("second"):
            conv2 = BN_RL_Conv(conv1, [output_channel, output_channel], [3, 3], [1, 1], is_training=is_training,
                               name=name)

        # 为了维度对齐
        if increase_dim:
            input_layer = Conv(input_layer, [input_channel, output_channel], [1, 1], [2, 2], padding="SAME", name=name)

        return conv2 + input_layer


# 下面是第二种残差块，当layer>50时我们就需要用这种
# 数据压缩又扩充
def bottle_residual_block(input_layer, output_channel, first_block=False, is_training=True, name=None):
    with tf.variable_scope(name):
        with tf.variable_scope("shortcut"):
            stride = None
            input_channel = input_layer.get_shape().as_list()[-1]
            if input_channel * 2 == output_channel:
                increase_dim = True
                stride = 2
            elif input_channel == output_channel:
                increase_dim = False
                stride = 1

            # 信息捷径
            input_layer = BN(input_layer, is_training=is_training)
            shortcut = RL(input_layer)
            shortcut = Conv(shortcut, [input_channel * 4, input_channel * 4], [1, 1], [stride, stride], name=name)

        with tf.variable_scope("first"):
            conv1 = None
            if first_block:
                conv1 = Conv(input_layer, [input_channel, output_channel], [1, 1], [1, 1], padding="SAME", name=name)
            else:
                conv1 = Conv_BN_RL(input_layer, [input_channel, output_channel], [1, 1], [1, 1], padding="SAME",
                                   is_training=is_training, name=name)

        with tf.variable_scope("second"):
            conv2 = Conv(conv1, [output_channel, output_channel], [3, 3], [stride, stride], name=name)

        with tf.variable_scope("third"):
            conv3 = BN_RL_Conv(conv2, [input_channel, input_channel * 4], [1, 1], [1, 1], is_training=is_training,
                               name=name)

        return conv3 + shortcut


def fully_connect(input_layer, units, weight_init=tf.contrib.layers.xavier_initializer(),
                  regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)):
    flatten = tf.layers.flatten(input_layer)
    input_layer = tf.layers.dense(input_layer, units=units, kernel_initializer=weight_init,
                                  kernel_regularizer=regularizer,
                                  use_bias=True)
    return input_layer


# 获取残差网络结构
def get_residual_layer(res_n):
    layerx = []
    if res_n == 18:
        layerx = [2, 2, 2, 2]
    if res_n == 34:
        layerx = [3, 4, 6, 3]
    if res_n == 50:
        layerx = [3, 4, 6, 3]
    if res_n == 101:
        layerx = [3, 4, 23, 3]
    if res_n == 152:
        layerx = [3, 8, 36, 3]
    return layerx


basic_channel = 32
# 根据layer选择不同的块结构
if Resnet_layer < 50:
    block = residual_block
else:
    block = bottle_residual_block

print("sys:数据输入处理")
# 通过cifar10_input脚本获取训练集与测试集
with tf.name_scope("get_data"):
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    """
    images_train.get_shape() = (100, 24, 24, 3)
    labels_train.get_shape() = (100,)
    """
    image_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=test_size)
    image_test, labels_test = cifar10_input.inputs(
        eval_data=True, data_dir=data_dir, batch_size=test_size)
print("sys:需要花一定时间收集数据。。。。")

# 构建图
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 3])
input_y = tf.placeholder(dtype=tf.int64, shape=[None])
training = tf.placeholder(tf.bool)

# 获得网络结构
residual_list = get_residual_layer(Resnet_layer)

# 构建网络
x = Conv(input_x, [input_x.get_shape().as_list()[-1], basic_channel], [3, 3], [1, 1], name="first_layer")

for i in range(residual_list[0]):
    x = block(x, basic_channel, True, training, "resblock0_%d" % i)

x = block(x, basic_channel * 2, is_training=training, name='resblock1_0')
for i in range(1, residual_list[1]):
    x = residual_block(x, basic_channel * 2, is_training=training, name='resblock1_%d' % i)

x = block(x, basic_channel * 4, is_training=training, name='resblock2_0')
for i in range(1, residual_list[2]):
    x = residual_block(x, basic_channel * 4, is_training=training, name='resblock2_%d' % i)

x = block(x, basic_channel * 8, is_training=training, name='resblock3_0')
for i in range(1, residual_list[3]):
    x = residual_block(x, basic_channel * 8, is_training=training, name='resblock3_%d' % i)

x = BN_RL(x, is_training=training)
x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
# 网络输出
x = fully_connect(x, units=class_count)
x = tf.reshape(x, [-1, 10])

# 计算误差
loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y, logits=x))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = loss + tf.reduce_sum(regu_losses)

with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#  计算准确率
prediction = tf.equal(tf.argmax(x, -1), input_y)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

print("sys:开始训练")
sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())  # 取代tf.global_variables_initializer()
tf.train.start_queue_runners(sess=sess)


########################### 测试区代码 ######################################

##########################################################################



start_time = 0
for step in range(training_step):
    image_batch, label_batch = sess.run([images_train, labels_train])
    train_op.run(feed_dict={input_x: image_batch, input_y: label_batch, training: True}, session=sess)
    if step == 0:
        start_time = time.time()
    if step % display_step == 0 and step != 0:
        now_time = time.time()
        cost_time = now_time - start_time
        feed_dict = {
            input_x: image_batch,
            input_y: label_batch,
            training: True
        }

        accuracy_eval, loss_eval = sess.run([accuracy, loss],
                                            feed_dict=feed_dict)
        print("step:%-8d accuracy=%-8g loss=%-8g time cost=%-8f(s)" % (step, accuracy_eval, loss_eval, cost_time))
        start_time = time.time()
end_time = time.time()
print("训练结束，所消耗的时间是 %d(s)" % (end_time - start_time))
print("平均每batch所消耗的时间是 %g(s)" % ((end_time - start_time) / (training_step / batch_size)))


image_batch, label_batch = sess.run([image_test, labels_test])
feed_dict = {
    input_x: image_batch,
    input_y: label_batch,
    training: False
}
test_accuracy = accuracy.eval(feed_dict=feed_dict, session=sess)
print("final accuracy {0}".format(test_accuracy))
