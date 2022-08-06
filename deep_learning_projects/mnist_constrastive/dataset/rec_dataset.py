import torchvision.datasets as datasets
import pathlib

path = pathlib.Path(__file__).parent.resolve()
dataset = datasets.MNIST(path, download=True)