from setuptools import setup

setup(
    name='wbl_model',
    version='0.1.4',
    packages=['wbl'],
    entry_points={
        'vllm.general_plugins': [
            "register_wbl_model = wbl:register",
        ],
    },
)
