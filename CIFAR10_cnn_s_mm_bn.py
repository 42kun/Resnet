import tensorflow as tf
import cifar10_input
import numpy as np
import time
import shutil

DELETE_BOARD = False
batch_size = 100
learning_rate = 1e-4
training_step = 30000
display_step = 100
record_step = 5000
train_size = 50000
test_size = 10000
# 数据集目录
data_dir = './CIFAR/'
print('begin')
with tf.name_scope("get_data"):
    # 获取训练集数据
    # images_train, labels_train = cifar10_input.inputs(
    #     eval_data=False, data_dir=data_dir, batch_size=batch_size)
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    #奇怪读取数据的过程移到下面程序就会无限制卡住
    image_test,labesl_test = cifar10_input.inputs(
        eval_data=True, data_dir=data_dir, batch_size=test_size)
    image_test_mid, labesl_test_mid = cifar10_input.inputs(
        eval_data=True, data_dir=data_dir, batch_size=1000)
print('begin data')




def weight_variable(name,shape,stddev=5e-2,wd=None):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev)
    if wd!=None:
        weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return tf.Variable(initial,name=name)

def bias_variable(name,shape,value):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial,name=name)

#输入参数
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 3])
input_y = tf.placeholder(dtype=tf.int32, shape=[None])
input_y_onehot = tf.placeholder(dtype=tf.int32, shape=[None,10])

#第一层卷积
with tf.name_scope("conv1"):
    W_conv1 = weight_variable("W_conv1",[5,5,3,64])
    b_conv1 = bias_variable("b_conv1",[64],0.0)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(input_x,W_conv1,[1,1,1,1],padding="SAME")+b_conv1)
    p_conv1 = tf.nn.max_pool(
        h_conv1,ksize = [1,3,3,1],strides = [1,2,2,1],padding="SAME",name = "p_conv1")
    n_conv1 = tf.layers.batch_normalization(p_conv1, momentum=0.9)


#第二层卷积
with tf.name_scope("conv2"):
    W_conv2 = weight_variable("W_conv2",[5,5,64,64])
    b_conv2 = bias_variable("b_conv2",[64],0.1)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(n_conv1,W_conv2,[1,1,1,1],padding="SAME")+b_conv2)
    n_conv2 = tf.layers.batch_normalization(h_conv2,momentum=0.9)
    p_conv2 = tf.nn.max_pool(
        n_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="p_conv2")


#全连接1
with tf.name_scope("local3"):
    p_conv2_length = int(p_conv2.shape[1]*p_conv2.shape[2]*p_conv2.shape[3])
    p_conv2_flat = tf.reshape(p_conv2,[-1,p_conv2_length])
    W_local3 = weight_variable("W_local3",[p_conv2_length,384],stddev=0.04,wd=0.004)
    b_local3 = bias_variable("b_local3",[384],0.1)
    h_local3 = tf.nn.relu(tf.matmul(p_conv2_flat,W_local3)+b_local3,name="h_local3")


#全连接2
with tf.name_scope("local4"):
    W_local4 = weight_variable("W_local4",[384,192],stddev=0.04,wd=0.004)
    b_local4 = bias_variable("b_local4",[192],0.1)
    h_local4 = tf.nn.relu(tf.matmul(h_local3,W_local4)+b_local4,name="h_local4")


#softmax
with tf.name_scope("softmmax"):
    W_soft = weight_variable("W_soft",[192,10],stddev=1/192.0)
    b_soft = bias_variable("b_soft",[10],0.0)
    h_soft = tf.add(tf.matmul(h_local4,W_soft),b_soft,name="h_soft")

#暂不是很清楚这个函数的意思
# _activation_summary(h_soft)

with tf.name_scope("loss"):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=input_y,logits = h_soft)
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name="cross_entropy")
    tf.add_to_collection("losses",cross_entropy_mean)
    loss = tf.add_n(tf.get_collection("losses"),name="total_loss")

#计算准确率
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.arg_max(input_y_onehot, 1), tf.arg_max(h_soft, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

with tf.name_scope("train"):
    opt_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

#绘制图表
timeTensor = tf.placeholder("float")
acc = tf.summary.scalar("accuracy",accuracy)
los = tf.summary.scalar("loss",loss)
tim = tf.summary.scalar("time cost",timeTensor)
tensorboard_dir = "logs/"
# if DELETE_BOARD:
#     shutil.rmtree(tensorboard_dir)
writer = tf.summary.FileWriter(tensorboard_dir,sess.graph)
merge_summary = tf.summary.merge_all()

start_time = time.time()
start_time_100 = 0
accuracy_dict = {}
print("start train")
for step in range(training_step):
    image_batch, label_batch = sess.run([images_train, labels_train])
    opt_op.run(feed_dict={input_x: image_batch, input_y: label_batch}, session=sess)
    if step==0:
        start_time_100 = time.time()
    if step%display_step==0 and step!=0:
        now_time = time.time()
        cost_time = now_time-start_time_100
        feed_dict = {
            input_x: image_batch,
            input_y: label_batch,
            input_y_onehot: np.eye(10, dtype=np.float32)[label_batch],
            timeTensor:cost_time
        }
        summary = sess.run(merge_summary,feed_dict=feed_dict)
        writer.add_summary(summary=summary,global_step=step)
        # print(sess.run(h_soft,feed_dict=feed_dict))
        # print(sess.run(input_y_onehot,feed_dict=feed_dict))
        accuracy_eval,loss_eval = sess.run([accuracy,loss],
            feed_dict=feed_dict)
        print("step:%-8d accuracy=%-8g loss=%-8g time cost=%-8f(s)"%(step,accuracy_eval,loss_eval,cost_time))
        if step%record_step==0:
            image_batch, label_batch = sess.run([image_test_mid, labesl_test_mid])
            feed_dict = {
                input_y_onehot: np.eye(10, dtype=np.float32)[label_batch],
                input_x: image_batch,
                input_y: label_batch
            }
            accuracy_eval = sess.run(accuracy,feed_dict=feed_dict)
            accuracy_dict[step] = accuracy_eval
            print(accuracy_dict)
        start_time_100 = time.time()
end_time = time.time()
print("train finish,time cost %d(s)"%(end_time-start_time))
print("avg time every batch %g(s)"%((end_time-start_time)/(training_step/batch_size)))

image_batch,label_batch = sess.run([image_test,labesl_test])
feed_dict = {
    input_y_onehot:np.eye(10, dtype=np.float32)[label_batch],
    input_x: image_batch,
    input_y: label_batch
}
test_accuracy = accuracy.eval(feed_dict=feed_dict, session=sess)
print("final accuracy {0}".format(test_accuracy))