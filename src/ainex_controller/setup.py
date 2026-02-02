from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ainex_controller'

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
    maintainer='wenlan',
    maintainer_email='wenlan.shen@tum.de',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ainex_hands_control_node = ainex_controller.ainex_hands_control_node:main',
            'human_to_ainex_basis = ainex_controller.human_to_ainex_basis:main',
            'ainex_imitation_control_node = ainex_controller.ainex_imitation_control_node:main',
            'project_cubes_onto_workspace = ainex_controller.project_cubes_onto_workspace:main',
            'stack_cubes = ainex_controller.stack_cubes:main',
            'stack_cubes_fixed = ainex_controller.stack_cubes_fixed:main',
        ],
    },
)
