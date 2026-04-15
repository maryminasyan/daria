from setuptools import setup, find_packages

setup(
    name="daria",
    version="1.0.0",
    author="Mary Minasyan",
    author_email="minasyan@caltech.edu",
    install_requires=[
        "numpy >= 1.24.3",
        "scipy >= 1.10.1",
        "astropy >= 5.3",
        "hmf",
        "dust_attenuation"
        ],
    packages=find_packages(),
    include_package_data=True
    )
