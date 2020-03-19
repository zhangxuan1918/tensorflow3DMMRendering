import os

import tensorflow as tf

from example_utils import load_images, load_params
from tf_3dmm.mesh.transform import affine_transform
import matplotlib.pyplot as plt

from tf_3dmm.morphable_model.morphable_model import TfMorphableModel


def compute_landmarks(bfm, shape_param, exp_param, pose_param, batch_size, resolution):
    vertices = bfm.get_vertices(shape_param=shape_param, exp_param=exp_param, batch_size=batch_size)
    transformed_vertices = affine_transform(
        vertices=vertices,
        scaling=pose_param[:, 0, 6:],
        angles_rad=pose_param[:, 0, 0:3],
        t3d=pose_param[:, 0:, 3:6]
    )
    lm_3d = tf.gather(transformed_vertices, bfm.kpt_ind, axis=1)
    lm_2d = tf.concat([tf.gather(lm_3d, [0], axis=2), resolution - tf.gather(lm_3d, [1], axis=2) - 1], axis=2)
    lm_2d = tf.transpose(lm_2d, perm=[0, 2, 1])
    return lm_2d.numpy()


def save_landmarks(images, landmarks, output_folder, resolution):
    n = len(images)
    fig = plt.figure()

    for i in range(n):

        filename = os.path.join(output_folder, str(i))
        im = images[i]
        lm = landmarks[i]

        ax = fig.add_subplot(1, n, i + 1)
        ax.set_ylim(bottom=resolution, top=0)
        ax.set_xlim(left=0, right=resolution)
        ax.imshow(im, origin='upper')
        if lm is not None:
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

        plt.savefig(filename)


if __name__ == '__main__':
    n_tex_para = 40
    bfm = TfMorphableModel(model_path='/opt/project/examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
    output_folder = '/opt/project/output/landmarks/landmark2'
    pic_names = ['image00002', 'IBUG_image_014_01_2', 'AFW_134212_1_0', 'IBUG_image_008_1_0']
    # pic_names = ['IBUG_image_008_1_0']
    batch_size = len(pic_names)
    images = load_images(pic_names, '/opt/project/examples/Data')
    resolution = 450
    shape_param, exp_param, _, _, _, pose_param = load_params(pic_names=pic_names, n_tex_para=n_tex_para,
                                                              data_folder='/opt/project/examples/Data/')
    landmarks = compute_landmarks(
        bfm=bfm, shape_param=shape_param, exp_param=exp_param, pose_param=pose_param, batch_size=batch_size,
        resolution=resolution)

    save_landmarks(images=images, landmarks=landmarks, output_folder=output_folder, resolution=resolution)
