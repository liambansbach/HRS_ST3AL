from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ainex_seq_model'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='niklas',
    maintainer_email='niklasderpeter@gmx.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'call_server = ainex_seq_model.call_server:main',
            'sequence_model_3D = ainex_seq_model.sequence_model_node:main',
            'sequence_model = ainex_seq_model.sequence_model_node:main',
        ],
    },
)
