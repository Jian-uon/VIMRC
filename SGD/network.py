#coding: utf-8
import numpy as np
import random

def sigmoid(z):#感知器函数
    return 1.0/(1.0+np.exp(-z))#np.exp(x)是e的x次方

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #第一层输入层不需要bias？ np.random.randn()是按照标准正太分布生成随机数 参数表示维度
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]#想了半天 zip其实就相当于矩阵转置。。

    def feedforward(self, a):# 将感知器公式用向量的形式表达a′=σ(wa+b)应用到每一层
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)#np.dot是内积，点乘的意思
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)# random.shuffle()打乱列表的顺序
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]#将训练数据按最小batch分组
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)# 学习！
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]#list.shape是获得当前维度下的宽度
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #开始计算梯度进行下降
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)#逆传播算法，计算delta
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation) #计算出所有的感知器的值
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])#先算最后一个的梯度
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())#transpose进行坐标轴交换，2维中是矩阵转置
        for l in xrange(2, self.num_layers):# 从后往前计算。。？？？
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): # 进行评价。。
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
