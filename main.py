import random
import tensorflow as tf
import numpy as np
from numpy.random import seed
from tensorflow.compat.v1.keras.datasets.fashion_mnist import load_data


(x_train, y_train), (x_test, y_test) = load_data()

n_train = len(x_train)
n_test = len(x_test)

n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_output = 10

seed(1)
tf.random.set_random_seed(seed = 2)

learning_rate = 1e-3
n_iterations = 1000
batch_size = 128
dropout = 0.2

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights ={
    'w1' : tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev = 0.1)),
    'w2' : tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev = 0.1)),
    'w3' : tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev = 0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1))
}

biases = {
    'b1' : tf.Variable(tf.constant(0.1, shape = [n_hidden1])),
    'b2' : tf.Variable(tf.constant(0.1, shape = [n_hidden2])),
    'b3' : tf.Variable(tf.constant(0.1, shape = [n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape = [n_output])),
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob = dropout)
output_layer = tf.add(tf.matmul(layer_3, weights['out']),biases['out'])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels = Y, logits = output_layer
    )
)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

x_test2 = np.reshape(np.array(x_test),(-1,784))
y_test2 = []
for i in range(len(y_test)):
    e = [0] * 10
    e[y_test[i]] = 1
    y_test2.append(e)

for n in range (n_iterations):
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        rand = random.randint(0, n_train-1)
        batch_x.append(np.reshape(np.array(x_train[rand]),(784)))
        
        num = y_train[rand]
        array = [0] * 10
        array[num] = 1
        batch_y.append(array)
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        reshaped_batch_x = np.reshape(np.array(batch_x),(-1,784))
        
    sess.run(train_step, feed_dict={X: reshaped_batch_x, Y: batch_y, keep_prob: dropout})

    if n % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X : reshaped_batch_x, Y : batch_y, keep_prob: 1.0})
        print(
                "Iteration",
                str(n),
                "\t| Loss =",
                str(minibatch_loss),
                "\t| Accuracy =",
                str(minibatch_accuracy)
                )

test_accuracy = sess.run(accuracy, feed_dict={X: x_test2, Y: y_test2, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)