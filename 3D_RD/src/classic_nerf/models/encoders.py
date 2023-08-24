import torch.nn as nn
import torch
import math 
class SinusoidalEncoder(nn.Module):
    """Sinusoidal positional encoder used in nerf"""


    def __init__(
            self , x_dim , min_deg ,max_deg ,use_identity : bool = True
    ):
        super().__init__()
        self.x_dim=x_dim
        self.min_deg=min_deg
        self.max_deg=max_deg
        self.use_identity=use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int :
        return (int(self.use_identity) + (self.max_deg - self.min_deg)*2)*self.x_dim
    
    def forward(self,x : torch.Tensor) -> torch.Tensor :
        """
        Args : 
            x : [...,x_dim]
        Returns :
            latent: [...,latent_dim]
        """
        if self.max_deg == self.min_deg :
            return x 
        xb = torch.reshape(
            (x[...,None,:]*self.scales[:,None]),
            list(x.shape[:-1])+ [(self.max_deg - self.min_deg)*self.x_dim],
        )
        latent = torch.sin(torch.cat([xb,xb+0.5*math.pi],dim=-1))
        if self.use_identity :
            latent= torch.cat([x]+[latent],dim=-1)

        return latent