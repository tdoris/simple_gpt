from setuptools import setup, find_packages

setup(
    name="simple_gpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.27.0",
        "datasets>=2.11.0",
        "tqdm>=4.65.0",
        "wandb>=0.14.0",
    ],
    author="User",
    author_email="user@example.com",
    description="A simple transformer-based language model implementation",
)
