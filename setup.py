from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="serm",
    version="1.0.2",
    author="Md Tauhidul Islam",
    author_email="tauhid@stanford.edu",
    description="SERM is a high-performance data-driven gene expression recovery framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xinglab-ai/self-consistent-expression-recovery-machine",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
)
