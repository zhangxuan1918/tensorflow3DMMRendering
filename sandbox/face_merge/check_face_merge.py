from tf_3dmm.mesh.render import render
import matplotlib.pyplot as plt
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from scipy import io as sio
import numpy as np
import tensorflow as tf


def load_3dmm(pic_name):
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
    mat_data = sio.loadmat(mat_filename)
    sp = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
    ep = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)

    tp = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
    cp = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
    ip = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
    pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)

    return sp, ep, tp, cp, ip, pp


tf_bfm = TfMorphableModel('../../examples/Data/BFM/Out/BFM.mat')
image_size = 450
# --load mesh data
pic_name1 = 'IBUG_image_014_01_2'
pic_name2 = 'AFW_134212_1_0'

sp1, ep1, tp1, cp1, ip1, pp1 = load_3dmm(pic_name1)
sp2, ep2, tp2, cp2, ip2, pp2 = load_3dmm(pic_name2)

mix_ratio = 0.2

spm = sp1 * mix_ratio + sp2 * (1 - mix_ratio)
epm = ep1 * mix_ratio + ep2 * (1 - mix_ratio)

image1 = render(
    pose_param=pp1,
    shape_param=sp1,
    exp_param=ep1,
    tex_param=tp1,
    color_param=cp1,
    illum_param=ip1,
    frame_height=image_size,
    frame_width=image_size,
    tf_bfm=tf_bfm
)

image2 = render(
    pose_param=pp2,
    shape_param=sp2,
    exp_param=ep2,
    tex_param=tp2,
    color_param=cp2,
    illum_param=ip2,
    frame_height=image_size,
    frame_width=image_size,
    tf_bfm=tf_bfm
)

image3 = render(
    pose_param=pp1,
    shape_param=spm,
    exp_param=epm,
    tex_param=tp1,
    color_param=cp1,
    illum_param=ip1,
    frame_height=image_size,
    frame_width=image_size,
    tf_bfm=tf_bfm
)

images = [image1, image2, image3]
titles = ['im1', 'im2', 'im3_{0}'.format(mix_ratio)]
fig = plt.figure()
n = len(images)

for i in range(n):
    im = images[i].numpy().astype(np.uint8)
    t = titles[i]

    ax = fig.add_subplot(1, n, i + 1)
    ax.set_ylim(bottom=image_size, top=0)
    ax.set_xlim(left=0, right=image_size)
    ax.imshow(im)
    ax.set_title(t)
    plt.savefig('test.png')