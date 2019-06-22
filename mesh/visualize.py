from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_mesh(vertices, triangles, subplot=[1, 1, 1], title='mesh', el=90, az=-90, lwdt=.1, dist=6, color="grey"):
    '''
    plot the mesh
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    '''
    ax = plt.subplot(subplot[0], subplot[1], subplot[2], projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, lw=lwdt, color=color, alpha=1)
    ax.axis("off")
    ax.view_init(elev=el, azim=az)
    ax.dist = dist
    plt.title(title)


def render_and_save(
        original_image,
        bfm,
        shape_param,
        exp_param,
        tex_param,
        color_param,
        illum_param,
        pose_param,
        landmarks,
        h,
        w,
        file_to_save=None):

    # render images
    image = bfm.render_3dmm(
        shape_param=shape_param,
        exp_param=exp_param,
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        pose_param=pose_param,
        h=h,
        w=w
    )

    if file_to_save is not None:
        fig = plt.figure()

        # plot original image
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(original_image)

        # plot rendered image
        image = np.asarray(np.round(image), dtype=np.int).clip(0, 255)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(image)

        # plot merged image
        # remove original face and use rendered face
        image_mask = image > 0
        original_image_removed = np.ma.masked_array(original_image, mask=image_mask, fill_value=0)
        image_overlay = original_image_removed + image
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(image_overlay)

        ax3.plot(landmarks[0, 0:17], landmarks[1, 0:17], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 17:22], landmarks[1, 17:22], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 22:27], landmarks[1, 22:27], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 27:31], landmarks[1, 27:31], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 31:36], landmarks[1, 31:36], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 36:42], landmarks[1, 36:42], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 42:48], landmarks[1, 42:48], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 48:60], landmarks[1, 48:60], marker='o', markersize=2, linestyle='-', color='w', lw=2)
        ax3.plot(landmarks[0, 60:68], landmarks[1, 60:68], marker='o', markersize=2, linestyle='-', color='w', lw=2)

        plt.savefig(file_to_save)
    return image
