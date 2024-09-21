import numpy as np
import MVGCN
from config import configMultiGCNBrain
from Datapreprocess.Datapre import pnc_dataset, get_training_validation_test
import tensorflow as tf
from sklearn.metrics import r2_score
import time
from utility.utils import sparse_sim, s2l, get_num_params


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
    N_roi = data_emoid.shape[1]
    train_ind, test_ind, valid_ind = get_training_validation_test(N)
    data_train = dict()
    data_valid = dict()
    data_test = dict()
    data_train['emoid'] = data_emoid[train_ind, :, :]
    data_train['nback'] = data_nback[train_ind, :, :]
    label_train = label[train_ind, :]
    mm = np.mean(label_train)
    label_train -= mm
    print(len(label))
    data_valid['emoid'] = data_emoid[valid_ind, :, :]
    data_valid['nback'] = data_nback[valid_ind, :, :]
    label_valid = label[valid_ind, :] - mm
    data_test['emoid'] = data_emoid[test_ind, :, :]
    data_test['nback'] = data_nback[test_ind, :, :]
    label_test = label[test_ind, :] - mm
    L = dict()

    s_emoid = np.zeros((N, N_roi, N_roi))
    L_emoid = np.zeros((N, N_roi, N_roi))
    s_nback = np.zeros((N, N_roi, N_roi))
    L_nback = np.zeros((N, N_roi, N_roi))
    for i in range(N):
        s_1 = np.abs(np.corrcoef(data_emoid[i, :, :]))
        s_2 = np.abs(np.corrcoef(data_nback[i, :, :]))
        s_1[np.isnan(s_1)] = 0
        s_1 = sparse_sim(s_1, 20)
        L_1 = s2l(s_1, norm_state=1)
        s_2[np.isnan(s_2)] = 0
        s_2 = sparse_sim(s_2, 20)
        L_2 = s2l(s_2, norm_state=1)
        s_emoid[i, :, :] = s_1
        s_nback[i, :, :] = s_2
        L_emoid[i, :, :] = L_1
        L_nback[i, :, :] = L_2
    L['emoid_train'] = L_emoid[train_ind, :, :]
    L['nback_train'] = L_nback[train_ind, :, :]
    L['emoid_valid'] = L_emoid[valid_ind, :, :]
    L['nback_valid'] = L_nback[valid_ind, :, :]
    L['emoid_test'] = L_emoid[test_ind, :, :]
    L['nback_test'] = L_nback[test_ind, :, :]

    return data_train, label_train, data_valid, label_valid, data_test, label_test, L


def placeholder_inputs(N_roi, d1, d2):
    """Generate placeholder variables to represent the input tensors.
    Args:
        N_roi: The number of rois.
        d1, d2: the time steps
    Returns:
        placeholder
    """
    input_lap1 = tf.placeholder(tf.float32, shape=(None, N_roi, N_roi), name='lap')
    input_lap2 = tf.placeholder(tf.float32, shape=(None, N_roi, N_roi), name='lap')
    input_x1 = tf.placeholder(tf.float32, shape=(None, N_roi, d1), name="input_x1")
    input_x2 = tf.placeholder(tf.float32, shape=(None, N_roi, d2), name="input_x2")
    input_labels = tf.placeholder(tf.float32, shape=(None, 1), name='labels')
    input_s1 = tf.placeholder(tf.float32, shape=(None, None), name='sim1')
    input_s2 = tf.placeholder(tf.float32, shape=(None, None), name='sim2')
    input_S = tf.placeholder(tf.float32, shape=(None, None), name='sim_all')
    return input_lap1, input_lap2, input_x1, input_x2, input_labels, input_s1, input_s2, input_S


def fill_feed_dict_train(input_lap1, input_lap2, input_x1, input_x2, input_labels,
                         L1, L2, x1, x2, labels):
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
        input_lap1: np.float32(L1),
        input_lap2: np.float32(L2),
        input_x1: np.float32(x1),
        input_x2: np.float32(x2),
        input_labels: np.float32(labels)
    }

    return feed_dict


def fill_feed_dict_valid(input_lap1, input_lap2, input_x1, input_x2, input_labels,
                         L1, L2, x1, x2, labels):
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
        input_lap1: np.float32(L1),
        input_lap2: np.float32(L2),
        input_x1: np.float32(x1),
        input_x2: np.float32(x2),
        input_labels: np.float32(labels),
    }

    return feed_dict


def fill_feed_dict_test(input_lap1, input_lap2, input_x1, input_x2, input_labels,
                        L1, L2, x1, x2, labels):
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
        input_lap1: np.float32(L1),
        input_lap2: np.float32(L2),
        input_x1: np.float32(x1),
        input_x2: np.float32(x2),
        input_labels: np.float32(labels),
    }

    return feed_dict


def run_training():
    data_train, label_train, data_valid, label_valid, data_test, label_test, L = read_data_sets()
    N_roi, d1, d2 = data_train['emoid'].shape[1], data_train['emoid'].shape[2], data_train['nback'].shape[2]
    print(d1, d2)
    input_lap1, input_lap2, input_x1, input_x2, input_labels, input_s1, input_s2, input_S = placeholder_inputs(N_roi, d1, d2)
    # Forward propagation
    # build the graph
    logits = MVGCN.inference(input_x1=input_x1, input_x2=input_x2, input_lap1=input_lap1,
                                                 input_lap2=input_lap2,  l_sizes=configMultiGCNBrain.l_sizes,
                                                 input_dim1=d1, input_dim2=d2)
    # loss function
    loss = MVGCN.loss(labels=input_labels, logits=logits)
    # cc = GCNMultiBrain.corcef(labels=input_labels, logits=logits)
    mae = MVGCN.MAE(labels=input_labels, logits=logits)
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = MVGCN.training(loss=loss, learning_rate=configMultiGCNBrain.learning_rate)
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
    feed_dict_train = fill_feed_dict_train(input_lap1=input_lap1, input_lap2=input_lap2, input_x1=input_x1, input_x2=input_x2, input_labels=input_labels,
                                           L1=L['emoid_train'], L2=L['nback_train'], x1=data_train['emoid'], x2=data_train['nback'],
                                           labels=label_train)
    feed_dict_valid = fill_feed_dict_valid(input_lap1=input_lap1, input_lap2=input_lap2, input_x1=input_x1, input_x2=input_x2, input_labels=input_labels,
                                           L1=L['emoid_valid'], L2=L['nback_valid'], x1=data_valid['emoid'], x2=data_valid['nback'],  labels=label_valid)
    feed_dict_test = fill_feed_dict_valid(input_lap1=input_lap1, input_lap2=input_lap2, input_x1=input_x1, input_x2=input_x2, input_labels=input_labels,
                                           L1=L['emoid_test'], L2=L['nback_test'], x1=data_test['emoid'], x2=data_test['nback'],  labels=label_test)
    # Run the Op to initialize the variables.
    Result_RMSE = []
    Result_MAE = []
    Result_R2 = []
    print("Optimization Start!")
    sess.run(init)
    print('number of parameters {}'.format(get_num_params()))
    for step in range(configMultiGCNBrain.max_steps):

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
        if (step + 1) % 10 == 0 or (step + 1) == configMultiGCNBrain.max_steps:
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

            R2 = r2_score(np.squeeze(label_test).reshape(-1), np.squeeze(sess.run(logits, feed_dict=feed_dict_test)).reshape(-1))
            print('r2 score:', R2)
            Result_R2.append(R2)
        # Evaluate against the test set.
    print("Optimization Finished!")
    print('Retrive the result...')




if __name__ == '__main__':
    run_training()


