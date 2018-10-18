import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

current_directory   = os.path.dirname(os.path.realpath(__file__))

# download the MNIST data in <current directory>/MNIST_data/
#one_hot=True: one-hot-encoding, means only return the highet probability
mnist = input_data.read_data_sets(current_directory + "/MNIST_data/", one_hot=True)

# X is placeholder for 28 x 28 image data
X = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is a 10 element ventor, it is the predicted probability of each digit class, e.g. [0, 0, 0.12, 0, 0, 0, 0.98, 0, 0.1, 0]
y_ = tf.placeholder(tf.float32, [None, 10])


# define the parameters we want to train, since we will be changing these value as the model learns, let's define them as Variable
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define the model, the model (try to) return the actual y
y = tf.nn.softmax(tf.matmul(X, W) + b) # use softmax as activation function

# define the lost funtion - cross entropy
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# minimize the loss in each step of training in gradient decent
learn_rate = 1
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_function)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    batch_size = 100
    # evaluate how well the model does. comapre the highest probability in actual (y) and predicted(y_)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # argmax return the index with the largest value across axes of a tensor, tf.equal return true if matching, correct_predition points to a tensor with list the true and false
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast true to 1 and false to 0
    # perform training
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # get batch_size data, batch_xs is iamge, batch_ys is label of 0~9
        sess.run(train_step, feed_dict={X: batch_xs, y_:batch_ys})
        batch_loss, batch_accuracy = sess.run([loss_function, accuracy], feed_dict={X: batch_xs, y_: batch_ys})
        print(str(i), ":\t loss = ", str(batch_loss), "\t accuracy = ", str(batch_accuracy))

    test_accuracy = sess.run(accuracy, feed_dict={X:mnist.test.images, y_:mnist.test.labels})
    print("test accuracy: {0}%".format(test_accuracy * 100.0))
    sess.close()