from setuptools import find_packages, setup

setup(
    name="GibbsPreparationLearning",
    packages=find_packages(),
    version="0.1.0",
    description="Package for creating Gibbs states and learning their faulty hamiltonians",
    author="Marc Sanz Drudis",
    license="MIT",
    install_requires=["qiskit",
                      "tqdm",
                      "matplotlib",
                      "plotly",
                      "pandas",
                      ],
    dependency_links=["git+"],
)
