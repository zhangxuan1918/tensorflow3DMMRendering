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
