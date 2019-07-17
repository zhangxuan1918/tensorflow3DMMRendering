from tf_3dmm.mesh.transform import affine_transform
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from scipy import io as sio
import tensorflow as tf


tf_bfm = TfMorphableModel('../../examples/Data/BFM/Out/BFM.mat')
# --load mesh data
pic_name = 'IBUG_image_008_1_0'
# pic_name = 'IBUG_image_014_01_2'
mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
mat_data = sio.loadmat(mat_filename)
sp = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
ep = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)

vertices = tf_bfm.get_vertices(sp, ep)
triangles = tf_bfm.triangles

pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)
s = pp[0, 6]
angles = pp[0, 0:3]
t = pp[0, 3:6]

transformed_vertices = affine_transform(vertices, s, angles, t)

landmarks_raw = tf.gather_nd(transformed_vertices, tf.reshape(tf_bfm.get_landmark_indices(), (-1, 1)))
landmarks = tf.concat([tf.reshape(landmarks_raw[:, 0], (-1, 1)), 450 - tf.reshape(landmarks_raw[:, 1], (-1, 1)) - 1], axis=1)


print(landmarks.shape)
