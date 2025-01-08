from setuptools import find_packages, setup
import glob

package_name = 'ctrl'

setup(
    name=package_name,
    version='0.0.0',
    # packages=['ctrl'],
    packages=['ctrl', 'ctrl.ros2_MPC'],
    # packages=find_packages(include=['ctrl', 'ctrl.*'])
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='auto2204',
    maintainer_email='youweike@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'VehicleController = ctrl.VehicleController:main',
            'VehicleController_MPC = ctrl.VehicleController_MPC:main'
        ],
    },
)
