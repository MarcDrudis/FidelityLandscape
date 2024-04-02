from setuptools import find_packages, setup

setup(
    name="fidlib",
    packages=find_packages(),
    version="0.1.0",
    description="Python Package for Fidelity Landscape",
    author="Marc Sanz Drudis",
    license="MIT",
    install_requires=[
        "qiskit>=1",
        "tqdm",
        "matplotlib",
        "plotly",
        "pandas",
        "pylatexenc",
        "nbformat",
        "numba",
        "black",
        "isort",
        "joblib",
        "seaborn",
        # "windsurfer @ git+ssh:git@github.com:MarcDrudis/windsurfer.git#egg=windsurfer",
    ],
    dependency_links=["git+"],
)
