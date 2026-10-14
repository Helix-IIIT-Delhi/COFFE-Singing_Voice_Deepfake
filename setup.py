from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coffe-svdsa",
    version="1.0.0",
    author="Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Priyabrata Mallick, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma",
    description="COFFE: Singing Voice Deepfake Source Attribution with Chernoff Distance Fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helix-IIIT-Delhi/COFFE-Singing_Voice_Deepfake",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
