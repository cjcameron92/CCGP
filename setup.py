from setuptools import setup

setup(
    name="ccgp",
    packages=["ccgp"],
    version="0.0.2",
    description="A multigene gp system",
    author="Cameron Carvalho, Cole Corbett",
    author_email="cjcameron92@gmail.com",
    license="Apache2",
    url="https://github.com/cjcameron92/ccgp",
    install_requires=["numpy>=1.26.4", "scikit-learn>=1.4.2"],
    classifiers=[],
    python_requires=">=3",
)
