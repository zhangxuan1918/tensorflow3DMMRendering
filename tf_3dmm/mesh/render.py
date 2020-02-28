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

    return render_2(
        angles_grad=pose_param[0, 0:3],
        scaling=pose_param[0, 6],
        t3d=pose_param[0, 3:6],
        shape_param=shape_param,
        exp_param=exp_param,
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        frame_width=frame_width,
        frame_height=frame_height,
        tf_bfm=tf_bfm
    )


def render_2(
        angles_grad,
        scaling,
        t3d,
        shape_param,
        exp_param,
        tex_param,
        color_param,
        illum_param,
        frame_width: int,
        frame_height: int,
        tf_bfm: TfMorphableModel
):
    assert is_tf_expression(angles_grad)
    assert is_tf_expression(scaling)
    assert is_tf_expression(t3d)
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
        scaling=scaling,
        angles_rad=angles_grad,
        t3d=t3d
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
    pic_name = 'image00002'
    # pic_name = 'IBUG_image_014_01_2'
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
    mat_data = sio.loadmat(mat_filename)
    sp = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
    ep = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)

    tp = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
    cp = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
    ip = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
    pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)

    # image = render(
    #     pose_param=pp,
    #     shape_param=sp,
    #     exp_param=ep,
    #     tex_param=tp,
    #     color_param=cp,
    #     illum_param=ip,
    #     frame_height=450,
    #     frame_width=450,
    #     tf_bfm=tf_bfm
    # )

    image = render_2(
        angles_grad=pp[0, 0:3],
        t3d=pp[0, 3:6],
        scaling=pp[0, 6],
        shape_param=sp,
        exp_param=ep,
        tex_param=tp[:40, :],
        color_param=cp,
        illum_param=ip,
        frame_height=450,
        frame_width=450,
        tf_bfm=tf_bfm
    )

    import imageio
    import numpy as np
    imageio.imsave('./rendered_{0}.jpg'.format(pic_name), image.numpy().astype(np.uint8))