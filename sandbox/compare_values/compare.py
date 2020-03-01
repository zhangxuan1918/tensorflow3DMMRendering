from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
import tensorflow as tf
import scipy.io as sio
from dirt import lighting
# compare variables between original rendering code and batch rendering code for debugging

pic_name = 'IBUG_image_008_1_0'
mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)

n_tex_para = 40

mat_data = sio.loadmat(mat_filename)

shape_param = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
shape_param = tf.expand_dims(shape_param, 0)
exp_param = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)
exp_param = tf.expand_dims(exp_param, 0)
tex_param = tf.constant(mat_data['Tex_Para'][:n_tex_para, :], dtype=tf.float32)
tex_param = tf.expand_dims(tex_param, 0)
color_param = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
color_param = tf.expand_dims(color_param, 0)
illum_param = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
illum_param = tf.expand_dims(illum_param, 0)
pose_param = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)
pose_param = tf.expand_dims(pose_param, 0)

tf_bfm = TfMorphableModel(model_path='../../examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)

vertices = tf_bfm.get_vertices(
    shape_param=shape_param,
    exp_param=exp_param,
    batch_size=1
)

vertex_norm = lighting.vertex_normals(vertices, tf_bfm.triangles)
bvertex_colors = tf_bfm.get_vertex_colors(tex_param, color_param, illum_param, -vertex_norm, 1)
vertex_colors = tf_bfm.get_vertex_colors(tex_param[0], color_param[0], illum_param[0], -vertex_norm[0])

print('done')