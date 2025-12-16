from setuptools import setup

setup(
    name='wbl_bridge',
    version='1.0.0',
    packages=['wbl_bridge'],
    install_requires=['transformers>=4.51.0', 'torch>=2.5.0',],
    platforms='any',
)
