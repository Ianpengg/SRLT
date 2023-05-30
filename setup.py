from setuptools import setup, find_packages


setup(
    name='srlt',
    version='1.0.0',
    author='Ian Peng',
    author_email='ian01050@gmail.com',
    description='This is the package for labelling the Scanning Radar Data\'s segmentation mask',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        'opencv-python==4.2.0.32',
        'tqdm',
        'PyQt5==5.15.9'
    ],
   
)
