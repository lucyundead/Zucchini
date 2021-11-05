from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Zucchini',
    url='https://github.com/lucyundead/',
    author='Zhi Li',
    author_email='lucyundeadshao@gmail.com',
    # Needed to actually package something
    packages=['zucchini'],
    # Needed for dependencies
    install_requires=['numpy','scipy','h5py','astropy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A python3 package for doing Galactic Dynamics and simulations',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
