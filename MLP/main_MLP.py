import numpy as np
import MLP
from config import configMLP
from Datapreprocess.Datapre import pnc_dataset, get_training_validation_test
import tensorflow as tf
import time
from utility.utils import sparse_sim, s2l, get_num_params
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def read_data_sets():
    """
    read_data_set
    :return: data and label
    """
    PNC = pnc_dataset(fmri_log=[True, True, False], meta_log=[True, False])
    Data = PNC.return_roi2roi_network()
    data_emoid = Data['raw_fmri']['emoid']
    data_nback = Data['raw_fmri']['nback']
    label = Data['label']['WRAT']

    N = data_emoid.shape[0]
    data_emoid_corr = np.zeros((N, data_emoid.shape[1], data_emoid.shape[1]))
    data_nback_corr = np.zeros((N, data_nback.shape[1], data_nback.shape[1]))

    train_ind, test_ind, valid_ind = get_training_validation_test(N)
    N1, N2, N3 = len(train_ind), len(valid_ind), len(test_ind)
    print(N1, N2, N3)
    for i in range(N):
        data_emoid_corr[i, :, :] = np.corrcoef(data_emoid[i, :, :])
        data_nback_corr[i, :, :] = np.corrcoef(data_nback[i, :, :])
    data_train = dict()
    data_valid = dict()
    data_test = dict()
    data_train['emoid'] = data_emoid[train_ind, :, :].reshape((N1, -1))
    data_train['nback'] = data_nback[train_ind, :, :].reshape((N1, -1))
    data_valid['emoid'] = data_emoid[valid_ind, :, :].reshape((N2, -1))
    data_valid['nback'] = data_nback[valid_ind, :, :].reshape((N2, -1))
    data_test['emoid'] = data_emoid[test_ind, :, :].reshape((N3, -1))
    data_test['nback'] = data_nback[test_ind, :, :].reshape((N3, -1))

    # pca1 = PCA(n_components=0.99)
    # pca2 = PCA(n_components=0.99)
    # pca1.fit(data_train['emoid'])
    # pca2.fit(data_train['nback'])
    # data_train['emoid'] = pca1.transform(data_train['emoid'])
    # data_valid['emoid'] = pca1.transform(data_valid['emoid'])
    # data_test['emoid'] = pca1.transform(data_test['emoid'])
    # data_train['nback'] = pca2.transform(data_train['nback'])
    # data_valid['nback'] = pca2.transform(data_valid['nback'])
    # data_test['nback'] = pca2.transform(data_test['nback'])

    label_train = label[train_ind, :]
    mm = np.mean(label_train)
    label_train -= mm
    label_valid = label[valid_ind, :] - mm
    label_test = label[test_ind, :] - mm
    print(len(label))

    return data_train, label_train, data_valid, label_valid, data_test, label_test


def placeholder_inputs(d1, d2):
    """Generate placeholder variables to represent the input tensors.
    Args:
        N_roi: The number of rois.
        d1, d2: the time steps
    Returns:
        placeholder
    """

    input_x1 = tf.placeholder(tf.float32, shape=(None, d1), name="input_x1")
    input_x2 = tf.placeholder(tf.float32, shape=(None, d2), name="input_x2")
    input_labels = tf.placeholder(tf.float32, shape=(None, 1), name='labels')
    return input_x1, input_x2, input_labels


def fill_feed_dict_train(input_x1, input_x2, input_labels, x1, x2, labels):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        input_lap1, input_lap2, input_x1, input_x2, input_labels, input_s1, input_s2: placeholders
        L1, L2, x1, x2, labels, s1, s2: input data
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    # Get data set of images and labels for training,validation and testing.

    feed_dict = {
        input_x1: np.float32(x1),
        input_x2: np.float32(x2),
        input_labels: np.float32(labels)
    }

    return feed_dict


def fill_feed_dict_valid(input_x1, input_x2, input_labels, x1, x2, labels):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        input_lap1, input_lap2, input_x1, input_x2, input_labels: placeholders
        L1, L2, x1, x2, labels: input data
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    # Get data set of images and labels for training,validation and testing.

    feed_dict = {
        input_x1: np.float32(x1),
        input_x2: np.float32(x2),
        input_labels: np.float32(labels)
    }

    return feed_dict


def run_training():
    data_train, label_train, data_valid, label_valid, data_test, label_test= read_data_sets()
    d1, d2 = data_train['emoid'].shape[1], data_train['nback'].shape[1]
    print(d1, d2)
    input_x1, input_x2, input_labels = placeholder_inputs(d1, d2)
    # Forward propagation
    # build the graph
    logits = MLP.inference(input_x1=input_x1, input_x2=input_x2, input_dim1=d1, input_dim2=d2)
    # loss function
    loss = MLP.loss(logits=logits, labels=input_labels)
    mae = MLP.MAE(labels=input_labels, logits=logits)
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = MLP.training(loss=loss, learning_rate=configMLP.learning_rate)
    # Build the summary tensor
    # summary = tf.summary.merge_all()
    # Add the initialize op
    init = tf.global_variables_initializer()
    # Create a saver for writing training checkpoints.
    # saver = tf.train.Saver()
    # create session
    sess = tf.Session()
    # Instantiate a SummaryWriter to output summaries and the Graph.
    # summary_writer = tf.summary.FileWriter(configUniGCN.log_dir, sess.graph)

    # And then after everything is built:
    feed_dict_train = fill_feed_dict_train(input_x1, input_x2, input_labels, data_train['emoid'], data_train['nback'], label_train)
    feed_dict_valid = fill_feed_dict_valid(input_x1, input_x2, input_labels, data_valid['emoid'], data_valid['nback'],  label_valid)
    feed_dict_test = fill_feed_dict_valid(input_x1, input_x2, input_labels, data_test['emoid'], data_test['nback'],  label_test)
    # Run the Op to initialize the variables.
    Result_RMSE = []
    Result_MAE = []
    print("Optimization Start!")

    sess.run(init)
    print(get_num_params())
    for step in range(configMLP.max_steps):

        start_time = time.time()
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict_train)

        duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
        if step % 1 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            # summary_str = sess.run(summary, feed_dict_train)
            # summary_writer.add_summary(summary_str, step)
            # summary_writer.flush()
        if (step + 1) % 10 == 0 or (step + 1) == configMLP.max_steps:
            # checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            # saver.save(sess, checkpoint_file, global_step=step)

            # Evaluate against the validation set.
            print('Validation Data Eval:', sess.run(loss, feed_dict_valid))

            RMSE = sess.run(loss, feed_dict_test)
            print('Test Data Eval:', RMSE)
            Result_RMSE.append(np.sqrt(sess.run(loss, feed_dict_test)))

            # print('Test CCs:', sess.run(cc, feed_dict_test))
            MAE = sess.run(mae, feed_dict_test)
            print('Test MAEs:', MAE)
            Result_MAE.append(MAE)

        # Evaluate against the test set.
    print("Optimization Finished!")
    print('Retrive the result...')



if __name__ == '__main__':
    run_training()
