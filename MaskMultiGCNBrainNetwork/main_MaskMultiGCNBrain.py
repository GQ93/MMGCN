import os
import numpy as np
import MaskGCNMultiBrain
from config import configMaskMultiGCNBrain
from Datapreprocess.Datapre import pnc_dataset, get_training_validation_test
import tensorflow as tf
from sklearn.metrics import r2_score
import time
from utility.utils import sparse_sim, cross_modality_sim, s2l


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
    data_train = dict()
    data_valid = dict()
    data_test = dict()
    data_train['emoid'] = data_emoid
    data_train['nback'] = data_nback
    label_train = label
    mm = np.mean(label_train)
    label_train -= mm
    print(len(label))
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
    L['emoid_train'] = L_emoid
    L['nback_train'] = L_nback

    S_emoid = np.abs(np.corrcoef(np.reshape(s_emoid, (s_emoid.shape[0], -1))))
    S_nback = np.abs(np.corrcoef(np.reshape(s_nback, (s_nback.shape[0], -1))))
    S_emoid[np.isnan(S_emoid)] = 0
    S_nback[np.isnan(S_nback)] = 0
    S_cross_mod = cross_modality_sim(S_emoid, S_nback)
    S_emoid = s2l(S_emoid, norm_state=0)
    S_nback = s2l(S_nback, norm_state=0)
    S_cross_mod = s2l(S_cross_mod, norm_state=0)
    return data_train, label_train, L, S_emoid, S_nback, S_cross_mod


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


def fill_feed_dict_train(input_lap1, input_lap2, input_x1, input_x2, input_labels, input_s1, input_s2, input_S,
                         L1, L2, x1, x2, labels, s1, s2, S):
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
        input_labels: np.float32(labels),
        input_s1: np.float32(s1),
        input_s2: np.float32(s2),
        input_S: np.float32(S)
    }

    return feed_dict


def run_training():
    data_train, label_train, L, S_emoid, S_nback, S_cross_mod = read_data_sets()
    N_roi, d1, d2 = data_train['emoid'].shape[1], data_train['emoid'].shape[2], data_train['nback'].shape[2]
    print(d1, d2)
    input_lap1, input_lap2, input_x1, input_x2, input_labels, input_s1, input_s2, input_S = placeholder_inputs(N_roi, d1, d2)
    # Forward propagation
    # build the graph
    logits, z1_2, z2_2, vis, mask = MaskGCNMultiBrain.inference(input_x1=input_x1, input_x2=input_x2, input_lap1=input_lap1,
                                                 input_lap2=input_lap2,  l_sizes=configMaskMultiGCNBrain.l_sizes,
                                                 input_dim1=d1, input_dim2=d2)
    # loss function
    loss = MaskGCNMultiBrain.loss(labels=input_labels, logits=logits)
    mae = MaskGCNMultiBrain.MAE(labels=input_labels, logits=logits)
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = MaskGCNMultiBrain.training(loss=loss, z1_2=z1_2, z2_2=z2_2,
                                      s1=input_s1, s2=input_s2,
                                      S=input_S, learning_rate=configMaskMultiGCNBrain.learning_rate, mask=mask)
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
    feed_dict_train = fill_feed_dict_train(input_lap1=input_lap1, input_lap2=input_lap2, input_x1=input_x1, input_x2=input_x2,
                                           input_labels=input_labels, input_s1=input_s1, input_s2=input_s2, input_S=input_S,
                                           L1=L['emoid_train'], L2=L['nback_train'], x1=data_train['emoid'], x2=data_train['nback'],
                                           labels=label_train, s1=S_emoid, s2=S_nback, S=S_cross_mod)
    # Run the Op to initialize the variables.
    Result_RMSE = []
    Result_MAE = []
    Final_mask = []
    print("Optimization Start!")
    sess.run(init)
    for step in range(configMaskMultiGCNBrain.max_steps):

        start_time = time.time()
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict_train)

        duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
        if step % 10 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            # summary_str = sess.run(summary, feed_dict_train)
            # summary_writer.add_summary(summary_str, step)
            # summary_writer.flush()
        if (step + 1) % 500 == 0 or (step + 1) == configMaskMultiGCNBrain.max_steps:


            RMSE = sess.run(loss, feed_dict_train)
            print('RMSE:', RMSE)
            Result_RMSE.append(np.sqrt(sess.run(loss, feed_dict_train)))

            # print('Test CCs:', sess.run(cc, feed_dict_test))
            MAE = sess.run(mae, feed_dict_train)
            print('MAE:', MAE)
            Result_MAE.append(MAE)
            Final_mask.append(sess.run(mask))

    # Evaluate against the test set.
    print("Optimization Finished!")
    print('Retrive the result...')
    Final_mask = np.asarray(Final_mask)
    print(Final_mask.shape)
    print(Final_mask)

    # save mask
    from time import strftime
    now = strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    mask_file_name = now+"mask"+str(configMaskMultiGCNBrain.scale_l1)+".npy"
    np.save(os.path.join(r"F:\projects\MultiGCN\result\mask0", mask_file_name), Final_mask)







if __name__ == '__main__':
    run_training()


