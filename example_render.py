import os

import numpy as np
import matplotlib.pyplot as plt
from example_utils import load_params, load_images, plot_image_w_lm
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel


def example_render_batch(pic_names: list, tf_bfm: TfMorphableModel, save_to_folder: str, n_tex_para: int):
    batch_size = len(pic_names)
    images_orignal = load_images(pic_names, '/opt/project/examples/Data/300W_LP/')

    # pose_param: [batch, 1, n_pose_param]
    # shape_param: [batch, n_shape_para, 1]
    # exp_param:   [batch, n_exp_para, 1]
    # tex_param: [batch, n_tex_para, 1]
    # color_param: [batch, 1, n_color_para]
    # illum_param: [batch, 1, n_illum_para]

    shape_param_batch, exp_param_batch, tex_param_batch, color_param_batch, illum_param_batch, pose_param_batch, lm_batch = \
        load_params(pic_names=pic_names, n_tex_para=n_tex_para)
    lm_rended = tf_bfm.get_landmarks(shape_param_batch, exp_param_batch, pose_param_batch, batch_size, 450, is_2d=True,
                                     is_plot=True)

    images_rendered = render_batch(
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
    ).numpy().astype(np.uint8)

    for i, pic_name in enumerate(pic_names):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plot_image_w_lm(ax, 450, images_orignal[i], lm_batch[i])
        ax = fig.add_subplot(1, 2, 2)
        plot_image_w_lm(ax, 450, images_rendered[i], lm_rended[i])
        plt.savefig(os.path.join(save_to_folder, pic_name))


if __name__ == '__main__':
    n_tex_para = 40
    tf_bfm = TfMorphableModel(model_path='./examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
    save_rendered_to = './output/render_batch/'
    pic_names = ['IBUG_image_014_01_2', 'AFW_134212_1_0', 'IBUG_image_008_1_0']
    example_render_batch(pic_names=pic_names, tf_bfm=tf_bfm, save_to_folder=save_rendered_to, n_tex_para=n_tex_para)
