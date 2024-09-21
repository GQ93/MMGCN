import tensorflow as tf
from utility import models as ly
from utility import layers
from config import configMultiGCNBrain


def in_class_penalty(z, sub_lap):
    """
    laplacian penalty
    :param z: the output of gcn layer N by d
    :param sub_lap: N by N laplacian matrix
    :return: the regularization term
    """
    p = configMultiGCNBrain.scale1_of_penalty * tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(z), sub_lap), z))
    return p


def bt_class_penalty(z1, z2, sub_laps):
    """
    between class regularization penalty
    :param z1: the output of first modality gcn layer N1 by d
    :param z2: the output of second modalty gcn layer N2 by d
    :param sub_laps: N1+N2 by N1+N2 laplacian matrix
    :return: between class penalty
    """
    z = tf.concat([z1, z2], 0) # (N1+N2, d)
    p = configMultiGCNBrain.scale2_of_penalty * tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(z), sub_laps), z))
    return p


def inference(input_x1, input_x2, input_lap1, input_lap2,  l_sizes, input_dim1, input_dim2):
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
    z1_1, z2_1 = ly.MultiGraphConv(x1=input_x1, lap1=input_lap1, input_dim1=input_dim1, x2=input_x2,
                                 lap2=input_lap2, input_dim2=input_dim2, l_sizes=l_sizes)
    z1_2 = tf.reshape(z1_1, [-1, l_sizes[-1]*264])
    z2_2 = tf.reshape(z2_1, [-1, l_sizes[-1]*264])
    z2 = tf.concat([z1_2, z2_2], 1)
    z3 = layers.DenseLayer(input_dim=l_sizes[-1]*264*2, output_dim=1024, name='dense_layer1', regular=True,
                           activation=tf.nn.relu)(x=z2)
    z4 = layers.DenseLayer(input_dim=1024, output_dim=2048, name='dense_layer2', regular=True,
                           activation=tf.nn.relu)(x=z3)
    logits = layers.DenseLayer(input_dim=2048, output_dim=1, name='dense_layer3', regular=False,
                               activation=None)(x=z4)
    vis = dict()
    vis['gradient'] = tf.nn.relu(tf.reduce_mean(tf.gradients(logits, [z1_1, z2_1]), 2))
    vis['z1'] = z1_1
    vis['z2'] = z2_1
    return logits, z1_2, z2_2, vis


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        labels: Labels tensor,  [None, 1].
        logits: Logits tensor, float - [None, 1]
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


# def corcef(logits, labels):
#     """Calculates the loss from the logits and the labels.
#     Args:
#         labels: Labels tensor,  [None, 1].
#         logits: Logits tensor, float - [None, 1]
#     Return: correlation coefficient of type float
#     """
    # import tensorflow_probability as tfp
    # return tfp.stats.correlation(logits, labels)

def training(loss, z1_2, z2_2, s1, s2, S, learning_rate):
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
    regular_in_class1 = in_class_penalty(z=z1_2, sub_lap=s1)
    regular_in_class2 = in_class_penalty(z=z2_2, sub_lap=s2)
    regular_bt_class = bt_class_penalty(z1=z1_2, z2=z2_2, sub_laps=S)
    loss += regular_in_class1 + regular_in_class2 + regular_bt_class
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
