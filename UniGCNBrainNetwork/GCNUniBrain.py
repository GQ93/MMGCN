import tensorflow as tf
from utility import models as ly
from utility import layers
from config import configUniGCNBrain


def lap_penalty(z, input_all_lap):
    """
    laplacian penalty
    :param z: the output of gcn layer N by d
    :param input_all_lap: N by N laplacian matrix
    :return: the regularization term
    """
    p = configUniGCNBrain.scale_of_penalty * tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(z), input_all_lap), z))
    return p


def inference(input_x, input_lap, l_sizes, input_dim):
    """build the graph
    Args:
        input_x: placeholder (batch_size, n_roi, input_dim)
        input_lap: placeholder (batch_size, n_roi, n_roi)
        input_dim: original feature dimension
        l_sizes: size of layers
    Returns:
        logits and penalty
    """
    z1 = ly.UniSemiGraphConv(x=input_x, lap=input_lap, input_dim=input_dim, l_sizes=l_sizes)
    z2 = tf.reshape(z1, [-1, l_sizes[-1]*264])
    z3 = layers.DenseLayer(input_dim=l_sizes[-1]*264, output_dim=1024, name='dense_layer1', regular=True,
                           activation=tf.nn.relu)(x=z2)
    logits = layers.DenseLayer(input_dim=1024, output_dim=1, name='dense_layer2', regular=False,
                               activation=None)(x=z3)
    return logits, z2


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
        penalty: laplacian penalty
    Return: Loss tensor of type float
    """
    labels = tf.cast(labels, dtype=tf.float32)
    with tf.name_scope('loss'):
        cost = tf.pow(logits-labels, 2)
    return tf.reduce_mean(cost)



def MAE(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
    Return: Loss tensor of type float
    """
    return tf.reduce_mean(tf.abs(logits - labels))


def training(loss, z, input_all_lap, learning_rate):
    """
    Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    penalty = lap_penalty(z=z, input_all_lap=input_all_lap)
    loss += penalty
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
