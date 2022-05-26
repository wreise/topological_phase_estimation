import setuptools

setuptools.setup(
    name="segmentation",
    version="0.1",
    author="Wojciech Reise",
    author_email="wojciech.reise@inria.fr",
    description="Topological phase estimation",
    long_description="Implementation of the methods presented in `Topological Phase Estimation`",
    url="https://github.com/wreise/topological_phase_estimation",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.19.4",
        "scipy>=1.5.4",
        "gudhi>=3.3.0",
        "scikit-learn>=0.23.2"],
    extras_require={
        "experiments": ["tqdm>=4.54.0", "pandas"],
        "notebooks": ["tqdm>=4.54.0", "matplotlib>=3.3.3", "plotly>=4.10.0", "jupyter"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
