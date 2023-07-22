from setuptools import setup

descr = """Borrelia_Cell_Segmentation: a cell segmentation pipeline for Borrelia cells in python.
This is meant to apply a simple otsu threshold to all images, identify the masks, and sort them into a Pandas dataframe.
Made by Joshua McCausland in the CJW lab, 2023.
"""

setup(
    name = 'Borrelia_Cell_Segmentation',
    version = '1.1',
    author = 'J. McCausland',
    author_email= 'jmccaus@stanford.edu',
    long_description=descr,
    packages = ['Borrelia_Cell_Segmentation'],
    install_requires = [
        'numpy',
        'pandas',
        'sknw_jwm @ git+https://github.com/jwmcca/sknw_edit.git',
        'scikit-image',
        'scipy',
        'opencv-python',
        'nd2reader'
        ], 
    )