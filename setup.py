#!/usr/bin/env python

from distutils.core import setup

setup(
    name='MCEq',
    version='0.99',
    description='Numerical cascade equation solver',
    author='Anatoli Fedynitch',
    author_email='afedynitch@gmail.com',
    url='https://github.com/afedynitch/MCEq.git',
    packages=[
        'MCEq',
    ],
    py_modules=['mceq_config'],
    requires=['numpy', 'scipy', 'numba'])