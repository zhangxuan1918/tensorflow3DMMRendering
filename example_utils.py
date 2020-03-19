import os
import numpy as np
import PIL
import tensorflow as tf
import scipy.io as sio


def load_params(pic_names, n_tex_para, data_folder='./examples/Data/'):

    # --load mesh data
    shape_param_batch = []
    exp_param_batch = []
    tex_param_batch = []
    color_param_batch = []
    illum_param_batch = []
    pose_param_batch = []

    for pic_name in pic_names:
        mat_filename = os.path.join(data_folder, '{0}.mat'.format(pic_name))
        mat_data = sio.loadmat(mat_filename)

        shape_param_batch.append(tf.constant(mat_data['Shape_Para'], dtype=tf.float32))
        exp_param_batch.append(tf.constant(mat_data['Exp_Para'], dtype=tf.float32))
        tex_param_batch.append(tf.constant(mat_data['Tex_Para'][:n_tex_para, :], dtype=tf.float32))
        color_param_batch.append(tf.constant(mat_data['Color_Para'], dtype=tf.float32))
        illum_param_batch.append(tf.constant(mat_data['Illum_Para'], dtype=tf.float32))
        pose_param_batch.append(tf.constant(mat_data['Pose_Para'], dtype=tf.float32))

    shape_param_batch = tf.stack(shape_param_batch, axis=0)
    exp_param_batch = tf.stack(exp_param_batch, axis=0)
    tex_param_batch = tf.stack(tex_param_batch, axis=0)
    color_param_batch = tf.stack(color_param_batch, axis=0)
    illum_param_batch = tf.stack(illum_param_batch, axis=0)
    pose_param_batch = tf.stack(pose_param_batch, axis=0)

    return shape_param_batch, exp_param_batch, tex_param_batch, color_param_batch, illum_param_batch, pose_param_batch


def load_images(pic_names, folder):
    images = []

    for pic_name in pic_names:
        image_filename = os.path.join(folder, '{0}.jpg'.format(pic_name))
        with open(image_filename, 'rb') as file:
            img = PIL.Image.open(image_filename)
            img = np.asarray(img)
            images.append(img)

    return images