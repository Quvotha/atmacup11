import torch
import torch.nn as nn
from typing import Tuple

import torch.nn as nn
from torchvision import models

from dataset import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_NUM_CHANNELS


def _flatten_image(batch: torch.Tensor) -> torch.Tensor:
    """Flatten given images.

    Parameters
    ----------
    batch : torch.Tensor
        Images to be flattened, shape = (batch_size, width, height, number of channels).

    Returns
    -------
    flattened
        Flattened images, shape = (batch_size, width * height * number of channels)
    """
    batch_size = batch.shape[0]
    return batch.view(batch_size, -1)


class AutoEncoderV01(nn.Module):
    """Fully-connected auto-encoder.
    """

    def __init__(self, input_dim: int = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_NUM_CHANNELS,
                 latent_space_dim: int = 64):
        assert(input_dim >= latent_space_dim > 0)
        super(AutoEncoderV01, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_space_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim, input_dim),
        )
        self.input_dim = input_dim
        self.latent_space_dim = latent_space_dim

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode given image and get lower dimension representaion.

        Parameters
        ----------
        batch : torch.Tensor
            Images flattened.

        Returns
        -------
        Encoded image: torch.Tensor
            shape = (batch_size, `self.latent_space_dim`)
        """
        return self.encoder(batch)

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct original images from given encoded images.

        Parameters
        ----------
        batch : torch.Tensor
            Encoded images.

        Returns
        -------
        Reconstructed image: torch.Tensor
            shape = (batch_size, `self.input_dim`)
        """
        return self.decoder(batch)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Parameters
        ----------
        batch : torch.Tensor
            Images, shape = (batch size, width, height, number of channels).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Flattened and normalized images, encoded images, reconstructed images.
        """
        batch_flattened = _flatten_image(batch)
        batch_flattened = batch_flattened / 255  # Normalization
        encoded = self.encode(batch_flattened)
        decoded = self.decode(encoded)
        return batch_flattened, encoded, decoded


def initialize_model(model_name: str, num_classes: int = 1) -> tuple:
    """Get model architecture.

    Parameters
    ----------
    model_name : str
        Should be one of 'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'.
    num_classes : int, optional
        [description], by default 1

    Returns
    -------
    (model, input_size): Tuple
        Model object and input image size.

    Refference
    ----------
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    model_ft = None
    input_size = 0

    if model_name == 'resnet':
        # Resnet18
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'alexnet':
        # Alexnet
        model_ft = models.alexnet(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vgg':
        # VGG11_bn
        model_ft = models.vgg11_bn(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'squeezenet':
        # Squeezenet
        model_ft = models.squeezenet1_0(pretrained=False)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == 'densenet':
        # Densenet
        model_ft = models.densenet121(pretrained=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'inception':
        # Inception v3
        # Be careful, expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(pretrained=False)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        raise ValueError(f'{model_name} is not valid model name')

    return model_ft, input_size
