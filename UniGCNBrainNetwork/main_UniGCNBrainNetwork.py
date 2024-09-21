import numpy as np
import GCNUniBrain
from config import configUniGCNBrain
from Datapreprocess.Datapre import pnc_dataset, get_training_validation_test
import tensorflow as tf
import time
from utility.utils import sparse_sim, get_num_params
from sklearn.metrics import r2_score

def read_data_sets():
    """
    read_data_set
    :return: data and label
    """
    PNC = pnc_dataset(fmri_log=[True, True, False], meta_log=[True, False])
    Data = PNC.return_roi2roi_network()
    data_emoid = Data['raw_fmri']['emoid']
    label = Data['label']['WRAT']
    N = data_emoid.shape[0]
    N_roi = data_emoid.shape[1]
    train_ind, test_ind, valid_ind = get_training_validation_test(N)
    data_train = data_emoid[train_ind, :, :]
    label_train = label[train_ind, :]
    mm = np.mean(label_train)
    label_train = label_train - mm
    data_valid = data_emoid[valid_ind, :, :]
    label_valid = label[valid_ind, :] - mm
    data_test = data_emoid[test_ind, :, :]
    label_test = label[test_ind, :] - mm
    L = np.zeros((N, N_roi, N_roi))
    for i in range(N):
        L_= np.corrcoef(data_emoid[i, :, :])
        L_[np.isnan(L_)] = 0
        L_ = sparse_sim(L_, 20)
        L[i, :, :] = L_
    L_train = L[train_ind, :, :]
    L_valid = L[valid_ind, :, :]
    L_test = L[test_ind, :, :]
    L_all_train = np.corrcoef(np.reshape(L_train, (L_train.shape[0], -1)))
    L_all_train[np.isnan(L_all_train)] = 0
    return data_train, label_train, L_train, data_valid, label_valid, L_valid, data_test, label_test, L_test, L_all_train


def placeholder_inputs(N_roi, d):
    """Generate placeholder variables to represent the input tensors.
    Args:
        N_roi: The number of rois.
        d: the time steps
    Returns:
        placeholder
    """
    input_lap = tf.placeholder(tf.float32, shape=(None, N_roi, N_roi), name='lap')
    input_x = tf.placeholder(tf.float32, shape=(None, N_roi, d), name="input_x")
    input_labels = tf.placeholder(tf.float32, shape=(None, 1), name='labels')
    input_all_lap = tf.placeholder(tf.float32, shape=(None, None), name='lap_all')
    return input_lap, input_x, input_labels, input_all_lap


def fill_feed_dict_train(input_lap, input_x, input_labels, input_all_lap, L, x, labels, L_all):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        input_lap, input_x, input_labels, input_mask: placeholders
        L, x, labels, mask: input data
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    # Get data set of images and labels for training,validation and testing.

    feed_dict = {
        input_lap: np.float32(L),
        input_x: np.float32(x),
        input_labels: np.float32(labels),
        input_all_lap: np.float32(L_all)
    }

    return feed_dict


def fill_feed_dict_valid(input_lap, input_x, input_labels, L, x, labels):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        input_lap, input_x, input_labels, input_mask: placeholders
        L, x, labels, mask: input data
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    # Get data set of images and labels for training,validation and testing.

    feed_dict = {
        input_lap: np.float32(L),
        input_x: np.float32(x),
        input_labels: np.float32(labels)
    }

    return feed_dict

def run_training():
    data_train, label_train, L_train, data_valid, label_valid, L_valid, data_test, label_test, L_test, L_all_train = read_data_sets()
    N_roi, d = data_train.shape[1], data_train.shape[2]
    input_lap, input_x, input_labels, input_all_lap = placeholder_inputs(N_roi, d)
    # Forward propagation
    # build the graph
    logits, z = GCNUniBrain.inference(input_x=input_x, input_lap=input_lap, l_sizes=configUniGCNBrain.l_sizes,
                                      input_dim=d)
    # loss function
    loss = GCNUniBrain.loss(labels=input_labels, logits=logits)
    mae = GCNUniBrain.MAE(labels=input_labels, logits=logits)
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = GCNUniBrain.training(loss=loss, z=z, input_all_lap=input_all_lap,
                                    learning_rate=configUniGCNBrain.learning_rate)
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
    feed_dict_train = fill_feed_dict_train(input_lap=input_lap, input_x=input_x, input_labels=input_labels,
                                           input_all_lap=input_all_lap, L=L_train, x=data_train,
                                           labels=label_train, L_all=L_all_train)
    feed_dict_valid = fill_feed_dict_valid(input_lap=input_lap, input_x=input_x, input_labels=input_labels,
                                           L=L_valid, x=data_valid, labels=label_valid)
    feed_dict_test = fill_feed_dict_valid(input_lap=input_lap, input_x=input_x, input_labels=input_labels,
                                          L=L_test, x=data_test, labels=label_test)
    # Run the Op to initialize the variables.
    print("Optimization Start!")
    Result_RMSE = []
    Result_MAE = []
    Result_R2 = []
    sess.run(init)
    print(get_num_params())
    for step in range(configUniGCNBrain.max_steps):

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
        if (step + 1) % 1 == 0 or (step + 1) == configUniGCNBrain.max_steps:
            # checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            # saver.save(sess, checkpoint_file, global_step=step)

            # Evaluate against the validation set.
            print('Validation Data Eval:', sess.run(loss, feed_dict_valid))
            RMSE = sess.run(loss, feed_dict_test)
            print('Test Data Eval:', RMSE)
            Result_RMSE.append(np.sqrt(sess.run(loss, feed_dict_test)))

            MAE = sess.run(mae, feed_dict_test)
            print('Test MAEs:', MAE)
            Result_MAE.append(MAE)
            R2 = r2_score(np.squeeze(label_test).reshape(-1), np.squeeze(sess.run(logits, feed_dict=feed_dict_test)).reshape(-1))
            print('r2 score:', R2)
            Result_R2.append(R2)

    print("Optimization Finished!")
    print('Test Data Eval:', sess.run(loss, feed_dict_test))


if __name__ == '__main__':
    run_training()

