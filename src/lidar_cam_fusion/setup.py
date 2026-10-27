import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lidar_cam_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haidar',
    maintainer_email='haidar@todo.todo',
    description='Node Fusi Sensor Kamera Logitech MX Brio + LiDAR Livox Mid-360 untuk Robot ABU Robocon 2027',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fusion_node = lidar_cam_fusion.fusion_node:main',
            'camera_node = lidar_cam_fusion.camera_node:main',
        ],
    },
)
