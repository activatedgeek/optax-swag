from setuptools import setup, find_packages

VERSION = '0.0.dev0'

with open('README.md') as f:
    README = f.read()

with open('requirements.txt') as f:
    requirements = f.read().split()

setup(
    name='optax-swag',
    description='Stochastic Weight Averaging for Optax',
    long_description=README,
    long_description_content_type='text/markdown',
    version=VERSION,
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(exclude=[
        'scripts',
        'scripts.*',
        'experiments',
        'experiments.*',
        'notebooks',
        'notebooks.*',
    ]),
    install_requires=requirements,
    extras_require={})
