import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.axes import Axes

from example_utils import load_params_80k, load_images
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel


def plot_image_w_lm(ax: Axes, resolution: int, image, lm):
    ax.set_ylim(bottom=resolution, top=0)
    ax.set_xlim(left=0, right=resolution)
    ax.imshow(image, origin='upper')
    ax.plot(lm[0, 0:17], lm[1, 0:17], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 17:22], lm[1, 17:22], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 22:27], lm[1, 22:27], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 27:31], lm[1, 27:31], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 31:36], lm[1, 31:36], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 36:42], lm[1, 36:42], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 42:48], lm[1, 42:48], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 48:60], lm[1, 48:60], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)
    ax.plot(lm[0, 60:68], lm[1, 60:68], marker='o', markersize=2, linestyle='-',
            color='w', lw=2)


def example_render_batch3(pic_names: list, tf_bfm: TfMorphableModel, n_tex_para: int, save_to_folder: str,
                          resolution: int):
    batch_size = len(pic_names)

    images_orignal = load_images(pic_names, '/opt/project/examples/Data/80k/')

    shape_param_batch, exp_param_batch, pose_param_batch = load_params_80k(pic_names=pic_names)
    shape_param = tf.squeeze(shape_param_batch)
    exp_param = tf.squeeze(exp_param_batch)
    pose_param = tf.squeeze(pose_param_batch)
    pose_param = tf.concat([pose_param[:, :-1], tf.constant(0.0, shape=(batch_size, 1), dtype=tf.float32), pose_param[:, -1:]], axis=1)
    lm = tf_bfm.get_landmarks(shape_param, exp_param, pose_param, batch_size, 450, is_2d=True, is_plot=True)

    images_rendered = render_batch(
        pose_param=pose_param,
        shape_param=shape_param,
        exp_param=exp_param,
        tex_param=tf.constant(0.0, shape=(len(pic_names), n_tex_para), dtype=tf.float32),
        color_param=None,
        illum_param=None,
        frame_height=450,
        frame_width=450,
        tf_bfm=tf_bfm,
        batch_size=batch_size
    ).numpy().astype(np.uint8)

    for i, pic_name in enumerate(pic_names):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plot_image_w_lm(ax, resolution, images_orignal[i], lm[i])
        ax = fig.add_subplot(1, 2, 2)
        plot_image_w_lm(ax, resolution, images_rendered[i], lm[i])
        plt.savefig(os.path.join(save_to_folder, pic_name))


if __name__ == '__main__':
    n_shape_para = 100
    n_tex_para = 40
    resolution = 450
    tf_bfm = TfMorphableModel(
        model_path='/opt/project/examples/Data/BFM/Out/BFM.mat',
        exp_path='/opt/project/examples/Data/BFM/Out/exp_80k.npz',
        n_shape_para=n_shape_para,
        n_tex_para=n_tex_para
    )
    save_rendered_to = './out'
    pic_names = ['afw_225191079_1_aug_29', 'helen_train_164866565_1_aug_28', 'ibug_image_007_aug_18',
                 'lfpw_train_image_0465_aug_11']
    example_render_batch3(pic_names=pic_names, tf_bfm=tf_bfm, n_tex_para=n_tex_para, save_to_folder=save_rendered_to, resolution=resolution)
