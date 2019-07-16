# Always prefer setuptools over distutils
import os
import re

from setuptools import setup
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def include_packages(root_packages_list):
    """
    This method generates a list of all package names to include
    starting from a list of root package names.
    :param root_packages_list: List of root package names to include
    :return: Returns the list of all package names to include
    """

    packages_to_include = list()

    for root_package in root_packages_list:
        for root, dirs, files in os.walk(root_package):
            if '__init__.py' in files:
                packages_to_include.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))

    return packages_to_include


setup(
    name='tensorflow3DMMRendering',
    version='0.0.1',
    description='Render 3DMM Model In Tensorflow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Xuan Zhang',
    author_email='zhangxuan1918@gmail.com',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Computer Vision',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3.7',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='rendering 3dmm tensorflow',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=include_packages(['tf_3dmm', 'sample']),  # Required
    python_requires='>=3.7',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['tensorflow-gpu>=2.0.0b1'],  # Optional
)