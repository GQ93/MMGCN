import tensorflow as tf
from utility import models as ly
from utility import layers
from config import configMTL

def inference(input_x1, input_x2, input_lap1, input_lap2, input_dim1, input_dim2):
    """build the graph
    Args:
        input_x1: placeholder (batch_size, n_roi, input_dim) of first modality
        input_x2: placeholder (batch_size, n_roi, input_dim) of second modality
        input_lap1: placeholder (batch_size, n_roi, n_roi) of first modality
        input_lap2: placeholder (batch_size, n_roi, n_roi) of second modality
        input_dim1: original feature dimension
        input_dim2: original feature dimension
        l_sizes: size of layers
    Returns:
        logits and penalty
    """
    z1, z2, l21w = ly.Multitasklearning(x1=input_x1, lap1=input_lap1, input_dim1=input_dim1, x2=input_x2, lap2=input_lap2, input_dim2=input_dim2)
    return z1, z2, l21w


def loss(z1, z2, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
    Return: Loss tensor of type float
    """
    labels = tf.cast(labels, dtype=tf.float32)
    with tf.name_scope('loss'):
        loss1 = tf.reduce_mean(tf.pow(z1-labels, 2))
        loss2 = tf.reduce_mean(tf.pow(z2-labels, 2))
        cost = 0.5*(loss1 + loss2)
    return cost, loss1, loss2


def MAE(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
    Return: Loss tensor of type float
    """
    return tf.reduce_mean(tf.abs(logits - labels))



def training(loss, l21w, learning_rate):
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

    loss += configMTL.scale_L21 * l21w
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op