import dirt
import tensorflow as tf
from dirt import lighting

from tf_3dmm.mesh.transform import affine_transform, affine_transform_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from tf_3dmm.tf_util import is_tf_expression


def render_batch(
        pose_param,
        shape_param,
        exp_param,
        tex_param,
        color_param,
        illum_param,
        frame_width: int,
        frame_height: int,
        tf_bfm: TfMorphableModel,
        batch_size: int
):
    assert is_tf_expression(pose_param)

    pose_shape = tf.shape(pose_param)
    if len(pose_shape) == 2:
        tf.debugging.assert_shapes([(pose_param, (batch_size, tf_bfm.get_num_pose_param()))])
        pose_param = tf.expand_dims(pose_param, 1)
    elif len(pose_shape) == 3:
        tf.debugging.assert_shapes([(pose_param, (batch_size, 1, tf_bfm.get_num_pose_param()))])
    else:
        raise ValueError('angles_grad shape wrong, dim != (batch, 1, 3) or (batch, 3)')

    vertices = tf_bfm.get_vertices_batch(shape_param=shape_param, exp_param=exp_param, batch_size=batch_size)
    vertex_norm = lighting.vertex_normals(vertices, tf_bfm.triangles)

    colors = tf_bfm.get_vertex_colors_batch(
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        vertex_norm=-vertex_norm,
        batch_size=batch_size
    )

    colors = tf.clip_by_value(colors / 255., 0., 1.)

    transformed_vertices = affine_transform_batch(
        vertices=vertices,
        scaling=pose_param[:, 0, 6],
        angles_rad=pose_param[:, 0, 0:3],
        t3d=pose_param[:, 0, 3:6]
    )
    transformed_vertices_x = transformed_vertices[:, :, 0] * 2 / frame_width - 1
    transformed_vertices_y = transformed_vertices[:, :, 1] * 2 / frame_height - 1
    transformed_vertices_z = -transformed_vertices[:, :, 2] / tf.reduce_max(tf.abs(transformed_vertices[:, :, 2]))

    # Convert vertices to homogeneous coordinates
    transformed_vertices = tf.concat([
        tf.expand_dims(transformed_vertices_x, axis=2),
        tf.expand_dims(transformed_vertices_y, axis=2),
        tf.expand_dims(transformed_vertices_z, axis=2),
        tf.ones_like(transformed_vertices[:, :, -1:])
    ], axis=2)

    # Render the G-buffer
    image = dirt.rasterise_batch(
        vertices=transformed_vertices,
        faces=tf.expand_dims(tf_bfm.triangles, axis=0),
        vertex_colors=colors,
        background=tf.zeros([1, frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    return image * 255


if __name__ == '__main__':
    import scipy.io as sio

    n_tex_para = 40
    tf_bfm = TfMorphableModel(model_path='../../examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)

    # --load mesh data
    pic_name = ['image00002', 'IBUG_image_014_01_2']
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
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

    image = render_batch(
        pose_param=pose_param,
        shape_param=shape_param,
        exp_param=exp_param,
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        frame_height=450,
        frame_width=450,
        tf_bfm=tf_bfm,
        batch_size=1
    )

    import imageio
    import numpy as np

    imageio.imsave('./rendered_{0}.jpg'.format(pic_name), image[0, :, :, :].numpy().astype(np.uint8))
