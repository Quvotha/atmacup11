import torch
import torch.nn as nn
from typing import Tuple

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
