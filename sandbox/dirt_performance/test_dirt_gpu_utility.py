from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
import scipy.io as sio
import tensorflow as tf

tf.debugging.set_log_device_placement(True)


n_tex_para = 40
tf_bfm = TfMorphableModel(model_path='/opt/project/examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
pic_names = ['image00002', 'IBUG_image_014_01_2', 'AFW_134212_1_0', 'IBUG_image_008_1_0'] * 8
batch_size = len(pic_names)


def my_load_params(pic_names, n_tex_para):

    # --load mesh data
    shape_param_batch = []
    exp_param_batch = []
    tex_param_batch = []
    color_param_batch = []
    illum_param_batch = []
    pose_param_batch = []

    for pic_name in pic_names:
        mat_filename = '/opt/project/examples/Data/{0}.mat'.format(pic_name)
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

shape_param_batch, exp_param_batch, tex_param_batch, color_param_batch, illum_param_batch, pose_param_batch = \
    my_load_params(pic_names=pic_names, n_tex_para=n_tex_para)
i = 0
while True:
    images = render_batch(
        pose_param=pose_param_batch,
        shape_param=shape_param_batch,
        exp_param=exp_param_batch,
        tex_param=tex_param_batch,
        color_param=color_param_batch,
        illum_param=illum_param_batch,
        frame_height=450,
        frame_width=450,
        tf_bfm=tf_bfm,
        batch_size=batch_size
    )
    i += 1
    print(i)