# -*- coding:utf-8 -*-
from setuptools import setup

setup(
    name='multivar_horner',
    packages=['multivar_horner'],
    description='python package implementing a multivariate Horner scheme for efficiently '
                'evaluating multivariate polynomials',
    author='J. Michelfeit',
    author_email='python@michelfe.it',
    license='MIT licence',
    license_file='LICENSE',
    url='https://github.com/MrMinimal64/multivar_horner',  # use the URL to the github repo
    project_urls={
        "Source Code": "https://github.com/MrMinimal64/multivar_horner",
        "Changelog": "https://github.com/MrMinimal64/multivar_horner/blob/master/CHANGELOG.rst",
    },
    keywords='math mathematics polynomial polynomials polynomial-evaluation multivariate multivariate-polynomials'
             ' horner horner-scheme factorisation factorization',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    long_description='Python package for computing Horner factorisations of multivariate polynomials '
                     'for efficient evaluation.\n'
                     'Please check GitHub for the documentation with plots: \n'
                     'https://github.com/MrMinimal64/multivar-horner',
    install_requires=[
        'numpy',
        'numba',
    ],
    python_requires='>=3.6',
    zip_safe=False,
    # TODO
    # extras_require={'pytorch': ["numba>=0.42"]},
)
