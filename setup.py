from setuptools import setup
  
setup(
    name='RemoveNoiseSpeech_WaveUnet_VietAE',
    packages=['Architecture, Tools'],
    install_requires=[
        'librosa',
        'wandb',
        'audiomentations',
        'tensorflow',
        'tensorflow-io',
        'keras',
        'pandas',
        'numpy'
    ],
)