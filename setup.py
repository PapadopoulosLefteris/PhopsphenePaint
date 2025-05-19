from setuptools import setup, find_packages

setup(
    name="PhosphenePaint",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "matplotlib",
        "dynaphos @ git+https://github.com/neuralcodinglab/dynaphos.git"  # if it's not on PyPI
    ],
    entry_points={
        "console_scripts": [
            "PhosphenePaint=main:main",  # Command-line script
        ]
    },
)
