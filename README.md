# tensorflow3DMMRendering

This project uses [dirt](https://github.com/pmh47/dirt) to render 3DMM model in tensorflow.

3DMM modified from [face3d](https://github.com/YadiraF/face3d)

## Dockerfile
 
### Tensorflow 2.0 && python 3.7

[dockerfile](dockerfiles/tf3_0_py3/Dockerfile)

### Tensorflow 1.3 && python 2.7

[dockerfile](dockerfiles/tf1_13_py2/Dockerfile)

## Example

Before running example

* put `BFM.mat` in folder `/examples/Data/BFM/Out/BFM.mat`. To generate `BFM.mat` see [face3d](https://github.com/YadiraF/face3d)
* put `IBUG_image_008_1_0.mat` to folder `/examples/Data/IBUG_image_008_1_0.mat`

### Tensorflow 2.0 && python 3.7

[example](sample/tensorflow_2_0_py3/texture_3dmm.py)

### Tensorflow 1.3 && python 2.7

[example](sample/tensorflow_1_13_py2/texture_3dmm.py)

