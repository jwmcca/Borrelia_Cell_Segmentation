from setuptools import setup

descr = """Borrelia_Cell_Segmentation: a cell segmentation pipeline for Borrelia cells in python.
This is meant to apply a simple otsu threshold to all images, identify the masks, and sort them into a Pandas dataframe.
Made by Joshua McCausland in the CJW lab, 2023.
"""

if __name__ == '__main__':
    setup('Borrelia_Cell_Segmentation',
          version = '1.0',
          url = descr,
          author = 'J. McCausland',
          author_email= 'jmccaus@stanford.edu',
          packages = ['borrelia_cell_segmentation'],
          install_requires = [
              'numpy',
              'pandas',
              'sknw_jwm @ git+https://github.com/jwmcca/sknw_edit.git',
              'scikit-image',
              'scipy',
              'opencv-python',
              'itertools',
              'nd2reader',
              'warnings',
              'os'
          ]
          )