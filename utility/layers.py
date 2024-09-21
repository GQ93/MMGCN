import tensorflow as tf


class GraphConvLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            regular=False,
            name="graph_conv"):
        """Initialise a Graph Convolution layer.

        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `graph_conv`.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name
        if regular:
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-7)
        else:
            regularizer = None
        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name='w',
                shape=(self.input_dim, self.output_dim),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=regularizer)

            if self.use_bias:
                self.b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x):
        x = tf.matmul(adj_norm, tf.matmul(x, self.w))  # AXW

        if self.use_bias:
            x = tf.add(x, self.use_bias)          # AXW + B

        if self.activation is not None:
            x = self.activation(x)                # activation(AXW + B)

        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

class MaskGraphConvLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            regular=False,
            name="graph_conv"):
        """Initialise a Graph Convolution layer.

        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `graph_conv`.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name
        if regular:
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-7)
        else:
            regularizer = None
        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name='w',
                shape=(self.input_dim, self.output_dim),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=regularizer)

            if self.use_bias:
                self.b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x, mask_h):
        diag_element = tf.eye(264)
        mask = tf.nn.relu(mask_h + tf.transpose(mask_h))
        x = tf.matmul(tf.multiply(mask+diag_element, adj_norm), tf.matmul(x, self.w))  # (Mask.*A_i)X_iW
        if self.use_bias:
            x = tf.add(x, self.use_bias)          # (Mask.*A_i)X_iW + B

        if self.activation is not None:
            x = self.activation(x)                # activation(Mask.*A_i)X_iW + B)

        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

class DenseLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=True,
            regular=False,
            name="dense"):
        """Initialise a dense layer.

        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `dense`.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name
        if regular:
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
        else:
            regularizer = None
        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name='w',
                shape=(self.input_dim, self.output_dim),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=regularizer)

            if self.use_bias:
                self.b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, x):
        x = tf.matmul(x, self.w)  # XW

        if self.use_bias:
            x = tf.add(x, self.use_bias)          # XW + B

        if self.activation is not None:
            x = self.activation(x)                # activation(XW + B)

        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


