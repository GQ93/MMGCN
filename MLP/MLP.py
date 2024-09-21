import tensorflow as tf
from utility import models as ly
from utility import layers
from config import configMTL

def inference(input_x1, input_x2, input_dim1, input_dim2):
    """build the graph
    Args:
        input_x1: placeholder (batch_size, input_dim) of first modality
        input_x2: placeholder (batch_size, input_dim) of second modality
        input_lap1: placeholder (batch_size, n_roi, n_roi) of first modality
        input_lap2: placeholder (batch_size, n_roi, n_roi) of second modality
        input_dim1: original feature dimension
        input_dim2: original feature dimension
        l_sizes: size of layers
    Returns:
        logits and penalty
    """
    input_x = tf.concat([input_x1, input_x2], 1)
    z1 = layers.DenseLayer(input_dim=input_dim1+ input_dim2, output_dim=1024, name='dense_layer1', regular=True,
                           activation=tf.nn.relu)(x=input_x)
    z2 = layers.DenseLayer(input_dim=1024, output_dim=2048, name='dense_layer2', regular=True,
                           activation=tf.nn.relu)(x=z1)
    logits = layers.DenseLayer(input_dim=2048, output_dim=1, name='dense_layer3', regular=False,
                               activation=None)(x=z2)

    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
    Return: Loss tensor of type float
    """
    labels = tf.cast(labels, dtype=tf.float32)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow(logits-labels, 2))
    return loss


def MAE(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
    Return: Loss tensor of type float
    """
    return tf.reduce_mean(tf.abs(logits - labels))



def training(loss, learning_rate):
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
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op