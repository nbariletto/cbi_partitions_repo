from setuptools import setup, find_packages

setup(
    name="cbi_partitions",
    version="0.1.0",
    description="Conformal Bayesian Inference for Partition-valued Data",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "numba", "tqdm"],
    python_requires='>=3.8',
)