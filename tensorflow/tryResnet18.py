import tensorflow as tf
import cifar10_input
import numpy as np
import time

# 生成权重
"""
stddev:权重标准差
wd:L2正则化参数，为了使权值尽可能小
"""
def weight_variable(shape,stddev=5e-2,wd=None,name=None):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev)
    if wd!=None:
        weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return tf.Variable(initial,name=name)

# 生成偏置
def bias_variable(shape,value,name=None):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial,name=name)





#bn层处理 #TODO:这里bn层参数基本都是用超参数的形式了，不参与训练（真的么？），但是我看一些教程还需要进行复杂设置。我认为需要去看一下（url1）
def BN(data,bn_momentum=0.9,name=None):
    return tf.layers.batch_normalization(data,momentum=bn_momentum,name = "%s_bn"%name)

#relu处理
def RL(data,name=None):
    return tf.nn.relu(data,name="%s_relu"%name)

#先将数据进行bn处理，再进行relu处理
def BN_RL(data,name):
    bn = BN(data,name)
    bn_rl = RL(bn)
    return bn_rl

#基础的卷积操作
"""
@data 输入的数据 tensor
@filter 输入维度与输出维度，一个1*2列表
@kernel 卷积核的形状，一个1*2列表
@stride 滑动步长，一个1*2元组
@pad 补零情况，字符串，可选"SAME"与"VALID"
@name
@wd 权重参数正则化参数
"""
def Conv(data,filter,kernel,stride=(1,1),padding="SAME",name=None,wd=None):
    shape = kernel+filter
    stride = [1,stride[0],stride[1],1]
    w = weight_variable(shape,wd=wd,name="%s_weight"%name)
    b = weight_variable(filter[-1:])
    return tf.nn.conv2d(data,w,stride,padding=padding,name = "%s_conv"%name) + b

#为卷积加上bn层
def Conv_BN(data,filter,kernel,stride=(1,1),padding="SAME",name=None,wd=None):
    cov = Conv(data,filter,kernel,stride,padding,name,wd)
    cov_bn = BN(cov,name="%s_bn"%name)
    return cov_bn

#为卷积加上bn层与RL层
def Conv_BN_RL(data,filter,kernel,stride=(1,1),padding="SAME",name=None,wd=None):
    cov_bn = Conv_BN(data,filter,kernel,stride,padding,name,wd)
    cov_bn_rl = RL(cov_bn)
    return cov_bn_rl


#池化层，包装了一下
"""
@data 输入的数据 tensor
@pool_type 池化类型 avg或者max
kernel 池化窗口大小 1*2元组
padding
stride 滑动步长 1*2元组
"""
def Pooling(data,pool_type="avg",kernel=(2,2),padding="VALID",stride=(2,2),name=None):
    ksize = [1,kernel[0],kernel[1],1]
    stride = [1,stride[0],stride[1],1]
    if pool_type=="avg":
        return tf.nn.avg_pool2d(data,ksize,stride,padding,name="%s_pool"%name)
    elif pool_type=="max":
        return tf.nn.max_pool2d(data,ksize,stride,padding,name="%s_pool"%name)
    else:
        raise Exception


BATCH_SIZE = 100
TEST_SIZE = 10000

data_dir = "../CIFAR/"

print("sys:数据输入处理")
# 通过cifar10_input脚本获取训练集与测试集
with tf.name_scope("get_data"):
    images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=BATCH_SIZE)
    image_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=TEST_SIZE)
print("sys:需要花一定时间收集数据。。。。")


input_x = tf.placeholder(dtype=tf.float32,shape=[None,24,24,3])
input_y = tf.placeholder(dtype=tf.int32,shape=[None])

with tf.name_scope("conv1"):
    conv1 = Conv_BN_RL(input_x,filter = (3,64),kernel=(7,7),stride=(2,2),padding="VALID",name="conv1")
    pool1 = Pooling(conv1,"max",kernel=(3,3),stride=(2,2),padding="VALID")
    print("que:",end="")
    print(conv1)
    print(pool1)

# with tf.name_scope("conv2"):
#     for i in range(2):


print("sys:开始训练")
sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer()) #取代tf.global_variables_initializer()

image_batch ,label_batch = sess.run([images_train,labels_train])
pool1.run(feed_dict={input_x:image_batch})

print("sys:程序结束")