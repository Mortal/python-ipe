from setuptools import setup, find_packages
from raster import DESCRIPTION


headline = DESCRIPTION.split('\n', 1)[0].rstrip('.')


setup(
    name='pygdal-raster',
    version='0.1',
    description=headline,
    long_description=DESCRIPTION,
    author='https://github.com/Mortal',
    url='https://github.com/Mortal/pygdal-raster',
    py_modules=['raster'],
    include_package_data=True,
    license='GPLv3',
    # entry_points={
    #     'console_scripts': ['pygdal-raster = raster:main'],
    # },
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
