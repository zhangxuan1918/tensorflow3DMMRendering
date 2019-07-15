import dirt
import tensorflow as tf
from dirt import lighting

from tf_3dmm.mesh.transform import affine_transform
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from tf_3dmm.tf_util import is_tf_expression


def render(
        pose_param,
        shape_param,
        exp_param,
        tex_param,
        color_param,
        illum_param,
        frame_width: int,
        frame_height: int,
        tf_bfm: TfMorphableModel
):
    assert is_tf_expression(pose_param)
    assert is_tf_expression(shape_param)
    assert is_tf_expression(exp_param)
    assert is_tf_expression(tex_param)
    assert is_tf_expression(color_param)
    assert is_tf_expression(illum_param)

    vertices = tf_bfm.get_vertices(shape_param=shape_param, exp_param=exp_param)
    vertex_norm = lighting.vertex_normals(vertices, tf_bfm.triangles)

    colors = tf_bfm.get_vertex_colors(
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        vertex_norm=-vertex_norm
    )

    colors = tf.clip_by_value(colors / 255., 0., 1.)

    transformed_vertices = affine_transform(
        vertices=vertices,
        scaling=pose_param[0, 6],
        angles_rad=pose_param[0, 0:3],
        t3d=pose_param[0, 3:6]
    )
    transformed_vertices_x = transformed_vertices[:, 0] * 2 / frame_width - 1
    transformed_vertices_y = transformed_vertices[:, 1] * 2 / frame_height - 1
    transformed_vertices_z = -transformed_vertices[:, 2] / tf.reduce_max(tf.abs(transformed_vertices[:, 2]))

    # Convert vertices to homogeneous coordinates
    transformed_vertices = tf.concat([
        tf.expand_dims(transformed_vertices_x, axis=1),
        tf.expand_dims(transformed_vertices_y, axis=1),
        tf.expand_dims(transformed_vertices_z, axis=1),
        tf.ones_like(transformed_vertices[:, -1:])
    ], axis=1)

    # Render the G-buffer
    image = dirt.rasterise(
        vertices=transformed_vertices,
        faces=tf_bfm.triangles,
        vertex_colors=colors,
        background=tf.zeros([frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    return image * 255


if __name__ == '__main__':
    import scipy.io as sio
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

    tp = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
    cp = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
    ip = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
    pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)

    image = render(
        pose_param=pp,
        shape_param=sp,
        exp_param=ep,
        tex_param=tp,
        color_param=cp,
        illum_param=ip,
        frame_height=450,
        frame_width=450,
        tf_bfm=tf_bfm
    )

    import imageio
    import numpy as np
    imageio.imsave('./textured_3dmm.jpg', image.numpy().astype(np.uint8))