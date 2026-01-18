from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'vision', 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.task')),
    ],
    install_requires=[
        'setuptools',
        ],
    zip_safe=True,
    maintainer='hrs2025',
    maintainer_email='tobias.toews@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_sub = vision.camera_sub:main',
            'detect_cubes = vision.detect_cubes:main',
            'mp_pose = vision.mp_upperbodypose_node:main',
            'test = vision.test:main',
            'identify_workspace = vision.identify_workspace:main',
        ],
    },
)
