import tensorflow as tf
from numpy import *
import time
import os
import numpy as np
from tensorflow.python.platform import gfile
start_time = time.time()
print('开始时间: ', time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
# ckpt = tf.train.latest_checkpoint('G:/result/Output_LeNet/')
# saver = tf.train.import_meta_graph('G:/result/Output_LeNet/model.ckpt-100.meta')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


train_ = np.load('G:/croprandom/gallery_data_256.npz')
test_ = np.load('G:/croprandom/probe_data_256.npz')

gallery_data = train_['images']
gallery_data_lbp = train_['lbps']
gallery_label = train_['label']

probe_data = test_['images']
probe_data_lbp = test_['lbps']
probe_label = test_['label']
# print(gallery_data)
#欧式距离
def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED

batch_size = 972

with tf.Session() as sess:
    with gfile.FastGFile('G:/result/S4_lenet/lenet1.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图
    sess.run(tf.global_variables_initializer())
    gallery_num = len(gallery_data)
    # saver.restore(sess, ckpt)
    input_x = tf.get_default_graph().get_tensor_by_name("x:0")
    input_z = tf.get_default_graph().get_tensor_by_name("z:0")
    output_fc2 = tf.get_default_graph().get_tensor_by_name("fc2:0")
    probe_output = np.zeros((11664,64), dtype=np.float32)
    gallery_output= np.zeros((11664, 64), dtype=np.float32)
    for i in range(0, len(probe_data) + 1, batch_size):
        start = i
        end = min(i + batch_size, len(probe_data))
        probe_data_a, probe_data_lbp_a = probe_data[start:end], probe_data_lbp[start:end]
        probe_output_fc2 = sess.run(output_fc2, feed_dict={input_x: probe_data_a, input_z: probe_data_lbp_a}) #p 特征
        probe_output[i:i+batch_size, :]=probe_output_fc2
    for i in range(0, len(gallery_data) + 1, batch_size):
        start = i
        end = min(i + batch_size, len(gallery_data))
        gallery_data_a, gallery_data_lbp_a = gallery_data[start:end], gallery_data_lbp[start:end]
        gallery_output_fc2 = sess.run(output_fc2,feed_dict={input_x: gallery_data_a, input_z: gallery_data_lbp_a}) #g特征
        gallery_output[i:i + batch_size, :] = gallery_output_fc2
    print(gallery_output)
    print(probe_output)
    end_time = time.time()
    print('结束时间: ', time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
    print('总耗时：', end_time - start_time, ' 秒')
    for i in range(gallery_num):
        dist = EuclideanDistances(gallery_output, probe_output)   #(972,972)  2个数据集的欧氏距离
        print(dist)
        print(dist.shape)
        sort_index = np.argsort(dist, axis=1) #从小到大返回dist的索引
        print('sort_index:', sort_index)
        print(sort_index.shape)
        sort_index_label = gallery_label[sort_index]#返回所有预测标签相对应的最小值的索引值的标签
        print('sort_index_label:', sort_index_label)
        print(sort_index_label.shape)
        predict_index = np.transpose(np.array(sort_index_label[::]))  #矩阵转至
        print(predict_index)
        actual_index = mat(probe_label)  #转成矩阵真实标签
        print(actual_index.shape)
        temp = np.cast['float32'](np.equal(actual_index, predict_index))
        print (temp)
        acc = []
        sum = 0
        i = 0
        while i < len(predict_index):
            sum = 0
            for j in range(len(predict_index)):
                if temp[i][j]==1:
                    sum += temp[i][j]
                    if i==(len(predict_index)-1):
                        cmc = sum / len(predict_index)

                    else:
                        temp[i+1][j]=temp[i][j]
            cmc=sum/len(predict_index)
            i+=1
            print(cmc)
            acc.append(cmc)

