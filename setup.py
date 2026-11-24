from setuptools import setup, find_packages

setup(
    name="aloha-bimanual-toss-catch",
    version="1.0.0",
    author="Hwihwa Lab",
    description="14-DOF Bimanual Robotic Dynamic Toss & Catch Gymnasium Environment",
    packages=find_packages(),
    package_data={
        "": ["assets/*.xml", "assets/*.png", "assets/*.gif"],
    },
    include_package_data=True,
    install_requires=[
        "mujoco>=3.1.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.8.0",
        "pillow>=9.5.0",
    ],
    python_requires=">=3.10",
)

