import tensorflow as tf
from layers import GraphConvLayer, DenseLayer, MaskGraphConvLayer
from config import configMaskMultiGCNBrain

def UniSemiGraphConv(x, lap, input_dim, l_sizes=None, para_name='modality1'):
    if l_sizes is None:
        l_sizes = [1024, 512, 512, 1]
    glayers = dict()
    with tf.name_scope(para_name):
        for i, _ in enumerate(l_sizes):
            name = 'graph_conv' + str(i+1)
            if i == 0:
                glayers[name] = GraphConvLayer(input_dim=input_dim, output_dim=l_sizes[i], name=name,
                                               regular=True, activation=tf.nn.sigmoid)(adj_norm=lap, x=x)
            elif i < len(l_sizes)-1:
                name_prev = 'graph_conv' + str(i)
                glayers[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name,
                                               regular=True, activation=tf.nn.sigmoid)(adj_norm=lap, x=glayers[name_prev])
            else:
                name_prev = 'graph_conv' + str(i)
                glayers[name] = GraphConvLayer(input_dim=l_sizes[i - 1], output_dim=l_sizes[i], name=name,
                                               regular=True, activation=tf.nn.relu)(adj_norm=lap, x=glayers[name_prev])
    return glayers[name]


def UniMLP(x, input_dim, l_sizes=None, para_name='modality1'):
    if l_sizes is None:
        l_sizes = [1024, 512, 512, 1]
    glayers = dict()
    with tf.name_scope(para_name):
        for i, _ in enumerate(l_sizes):
            name = 'mlp' + str(i+1)
            if i == 0:
                glayers[name] = DenseLayer(input_dim=input_dim, output_dim=l_sizes[i], name=name, regular=True,
                                           activation=tf.nn.relu)(x=x)
            elif i < len(l_sizes)-1:
                name_prev = 'mlp' + str(i)
                glayers[name] = DenseLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                           activation=tf.nn.relu)(x=glayers[name_prev])
            else:
                name_prev = 'mlp' + str(i)
                glayers[name] = DenseLayer(input_dim=l_sizes[i - 1], output_dim=l_sizes[i], name=name, regular=True,
                                           activation=tf.nn.relu)(x=glayers[name_prev])
    return glayers[name]


def MultiGraphConv(x1, lap1, input_dim1, x2, lap2, input_dim2, l_sizes=None):
    if l_sizes is None:
        l_sizes = [1024, 512, 512, 1]
    glayers1 = dict()
    glayers2 = dict()
    with tf.name_scope('modality1'):
        for i, _ in enumerate(l_sizes):
            name = 'graph1_conv' + str(i+1)
            if i == 0:
                glayers1[name] = GraphConvLayer(input_dim=input_dim1, output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap1, x=x1)
            elif i < len(l_sizes)-1:
                name_prev = 'graph1_conv' + str(i)
                glayers1[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap1, x=glayers1[name_prev])
            else:
                name_prev = 'graph1_conv' + str(i)
                glayers1[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.relu)(adj_norm=lap1, x=glayers1[name_prev])
        out1 = glayers1[name]
    with tf.name_scope('modality2'):
        for i, _ in enumerate(l_sizes):
            name = 'graph2_conv' + str(i+1)
            if i == 0:
                glayers2[name] = GraphConvLayer(input_dim=input_dim2, output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap2, x=x2)
            elif i < len(l_sizes)-1:
                name_prev = 'graph2_conv' + str(i)
                glayers2[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap2, x=glayers2[name_prev])
            else:
                name_prev = 'graph2_conv' + str(i)
                glayers2[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.relu)(adj_norm=lap2, x=glayers2[name_prev])
        out2 = glayers2[name]
    return out1, out2

def MVGraphConv(x1, lap1, input_dim1, x2, lap2, input_dim2, l_sizes=None):
    if l_sizes is None:
        l_sizes = [1024, 512, 512, 1]
    glayers1 = dict()
    glayers2 = dict()
    with tf.name_scope('modality1'):
        for i, _ in enumerate(l_sizes):
            name = 'graph1_conv' + str(i+1)
            if i == 0:
                glayers1[name] = GraphConvLayer(input_dim=input_dim1, output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap1, x=x1)
            elif i < len(l_sizes)-1:
                name_prev = 'graph1_conv' + str(i)
                glayers1[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap1, x=glayers1[name_prev])
            else:
                name_prev = 'graph1_conv' + str(i)
                glayers1[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.relu)(adj_norm=lap1, x=glayers1[name_prev])
        out1 = glayers1[name]
    with tf.name_scope('modality2'):
        for i, _ in enumerate(l_sizes):
            name = 'graph2_conv' + str(i+1)
            if i == 0:
                glayers2[name] = GraphConvLayer(input_dim=input_dim2, output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap2, x=x2)
            elif i < len(l_sizes)-1:
                name_prev = 'graph2_conv' + str(i)
                glayers2[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap2, x=glayers2[name_prev])
            else:
                name_prev = 'graph2_conv' + str(i)
                glayers2[name] = GraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.relu)(adj_norm=lap2, x=glayers2[name_prev])
        out2 = glayers2[name]
    return out1, out2

def L21regular(W):
    """
    L21 normalization for W
    :param W:
    :return:
    """
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.multiply(W, W), 1)))

def ManifoldMultitasklearning(x1, lap1, input_dim1, x2, lap2, input_dim2):
    W1 = tf.Variable(tf.random_normal([input_dim1, 1]))
    W2 = tf.Variable(tf.random_normal([input_dim2, 1]))

    z1 = tf.matmul(x1, W1)
    z2 = tf.matmul(x2, W2)
    r1 = tf.matmul(tf.matmul(tf.transpose(z1), lap1), z1)
    r2 = tf.matmul(tf.matmul(tf.transpose(z2), lap2), z2)
    l21w = L21regular(W1) + L21regular(W2)
    return z1, z2, r1, r2, l21w

def Multitasklearning(x1, lap1, input_dim1, x2, lap2, input_dim2):
    W1 = tf.Variable(tf.random_normal([input_dim1, 1]))
    W2 = tf.Variable(tf.random_normal([input_dim2, 1]))

    z1 = tf.matmul(x1, W1)
    z2 = tf.matmul(x2, W2)
    l21w = L21regular(W1) + L21regular(W2)
    return z1, z2, l21w

def Linearregression(x1, input_dim1):
    W1 = tf.Variable(tf.random_normal([input_dim1, 1]))
    B1 = tf.Variable(tf.random_normal([1, 1]))
    z1 = tf.matmul(x1, W1)
    return z1

def NewManifoldMultitasklearning(x1, input_dim1, x2, input_dim2, lap12):
    # lap12 shape(2,2)
    W1 = tf.Variable(tf.random_normal([input_dim1, 1]))
    W2 = tf.Variable(tf.random_normal([input_dim2, 1]))

    z1 = tf.matmul(x1, W1)
    z2 = tf.matmul(x2, W2)
    z = tf.concat([z1, z2], 0)# shape(2N, 1)
    r = tf.matmul(tf.matmul(tf.transpose(z), lap12), z)
    l21w = L21regular(W1) + L21regular(W2)
    return z1, z2, r, l21w

def MaskMultiGraphConv(x1, lap1, input_dim1, x2, lap2, input_dim2, l_sizes=None):
    if l_sizes is None:
        l_sizes = [1024, 512, 512, 1]
    mask_h = tf.get_variable(
        name='mask_h',
        shape=(configMaskMultiGCNBrain.mask_shape[0], configMaskMultiGCNBrain.mask_shape[1]),
        initializer=tf.zeros_initializer())
        # initializer=tf.glorot_uniform_initializer())
    glayers1 = dict()
    glayers2 = dict()
    with tf.name_scope('modality1'):
        for i, _ in enumerate(l_sizes):
            name = 'graph1_conv' + str(i+1)
            if i == 0:
                glayers1[name] = MaskGraphConvLayer(input_dim=input_dim1, output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap1, x=x1, mask_h=mask_h)
            elif i < len(l_sizes)-1:
                name_prev = 'graph1_conv' + str(i)
                glayers1[name] = MaskGraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap1, x=glayers1[name_prev], mask_h=mask_h)
            else:
                name_prev = 'graph1_conv' + str(i)
                glayers1[name] = MaskGraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.relu)(adj_norm=lap1, x=glayers1[name_prev], mask_h=mask_h)
        out1 = glayers1[name]
    with tf.name_scope('modality2'):
        for i, _ in enumerate(l_sizes):
            name = 'graph2_conv' + str(i+1)
            if i == 0:
                glayers2[name] = MaskGraphConvLayer(input_dim=input_dim2, output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap2, x=x2, mask_h=mask_h)
            elif i < len(l_sizes)-1:
                name_prev = 'graph2_conv' + str(i)
                glayers2[name] = MaskGraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.sigmoid)(adj_norm=lap2, x=glayers2[name_prev], mask_h=mask_h)
            else:
                name_prev = 'graph2_conv' + str(i)
                glayers2[name] = MaskGraphConvLayer(input_dim=l_sizes[i-1], output_dim=l_sizes[i], name=name, regular=True,
                                                activation=tf.nn.relu)(adj_norm=lap2, x=glayers2[name_prev], mask_h=mask_h)
        out2 = glayers2[name]
    mask = tf.nn.relu(mask_h + tf.transpose(mask_h))
    return out1, out2, mask
