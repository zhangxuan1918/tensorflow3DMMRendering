import dirt
import tensorflow as tf
import tensorflow_graphics as tfg

from tf_3dmm.mesh.transform import affine_transform
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
    """
    render faces in batch
    :param: pose_param: [batch, n_pose_para] or (batch, 1, n_pose_param)
    :param: shape_param: [batch, n_shape_para, 1] or [batch, n_shape_para]
    :param: exp_param:   [batch, n_exp_para, 1] or [batch, n_exp_para]
    :param: tex_param: [batch, n_tex_para, 1] or [batch, n_tex_para]
    :param: color_param: [batch, 1, n_color_para] or [batch, n_color_para]
    :param: illum_param: [batch, 1, n_illum_para] or [batch, n_illum_para]
    :param: frame_width: rendered image width
    :param: frame_height: rendered image height
    :param: tf_bfm: basel face model
    :param: batch_size: batch size
    :return: images, [batch, frame_width, frame_height, 3]
    """
    assert is_tf_expression(pose_param)

    pose_shape = tf.shape(pose_param)
    if pose_shape.shape[0] == 2:
        tf.debugging.assert_shapes(
            [(pose_param, (batch_size, tf_bfm.get_num_pose_param()))],
            message='pose_param shape wrong, dim != ({batch}, {dim})'.format(
                batch=batch_size, dim=tf_bfm.get_num_pose_param()))
        pose_param = tf.expand_dims(pose_param, 1)
    elif pose_shape.shape[0] == 3:
        tf.debugging.assert_shapes(
            [(pose_param, (batch_size, 1, tf_bfm.get_num_pose_param()))],
            message='pose_param shape wrong, dim != ({batch}, 1, {dim})'.format(
                batch=batch_size, dim=tf_bfm.get_num_pose_param()))
    else:
        raise ValueError('pose_param shape wrong, dim != ({batch}, 1, {dim}) or ({batch}, {dim})'.format(
                batch=batch_size, dim=tf_bfm.get_num_pose_param()))

    vertices = tf_bfm.get_vertices(shape_param=shape_param, exp_param=exp_param, batch_size=batch_size)
    # vertex_norm = lighting.vertex_normals(vertices, tf_bfm.triangles)
    vertex_norm = tfg.geometry.representation.mesh.normals.vertex_normals(
        vertices=vertices,
        indices=tf.repeat(tf.expand_dims(tf_bfm.triangles, 0), batch_size, axis=0),
        clockwise=True
    )

    colors = tf_bfm.get_vertex_colors(
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        vertex_norm=-vertex_norm,
        batch_size=batch_size
    )

    colors = tf.clip_by_value(colors / 255., 0., 1.)

    transformed_vertices = affine_transform(
        vertices=vertices,
        scaling=pose_param[:, 0, 6:],
        angles_rad=pose_param[:, 0, 0:3],
        t3d=pose_param[:, 0:, 3:6]
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
        faces=tf.tile(tf.expand_dims(tf_bfm.triangles, axis=0), (batch_size, 1, 1)),
        # faces=tf.expand_dims(tf_bfm.triangles, axis=0),
        vertex_colors=colors,
        background=tf.zeros([batch_size, frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    return image * 255


if __name__ == '__main__':
    import scipy.io as sio

    n_tex_para = 40
    tf_bfm = TfMorphableModel(model_path='../../examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)

    # --load mesh data
    pic_names = ['image00002', 'IBUG_image_014_01_2']
    shape_param_batch = []
    exp_param_batch = []
    tex_param_batch = []
    color_param_batch = []
    illum_param_batch = []
    pose_param_batch = []

    for pic_name in pic_names:
        mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
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

    batch_size = len(pic_names)

    image = render_batch(
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

    import imageio
    import numpy as np

    for i, pic_name in enumerate(pic_names):
        imageio.imsave('./rendered_{0}.jpg'.format(pic_name), image[i, :, :, :].numpy().astype(np.uint8))
