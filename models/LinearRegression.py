#!usr/bin/env python
# coding utf-8
'''
@File       :LinearRegression.py
@Copyright  :CV Group
@Date       :9/22/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf
import numpy as np

X = np.array([[2013, 2014, 2015, 2016, 2017]], dtype=np.float32)
Y = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
X = (X - X.min()) / (X.max() - X.min())
Y = Y / 10
# Y = (Y - Y.min()) / (Y.max() - Y.min())
w, b = 1, 1
learning_rate = 0.01
# for epoch in range(100):
#     y_pred = w * X + b
#     w_grad, b_grad = (Y - y_pred) @ X.transpose(), (Y - y_pred).sum()
#     w, b = w - learning_rate * w_grad, b - learning_rate * b_grad
#
# print(w, b)
x = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(Y)
w = tf.Variable(initial_value=1.)
b = tf.Variable(initial_value=1.)
num_epoch = 10000
lr = 5e-4
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
for epoch in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = x * w + b
        loss = tf.reduce_sum(tf.square(y - y_pred))
    # print(y_pred, loss)
    grads = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(grads_and_vars=zip(grads, [w, b]))
    print(loss)
    # w_grad, b_grad = tape.gradient(loss, [w, b])
    # w.assign_sub(lr * w_grad)
    # b.assign_sub(lr * b_grad)
