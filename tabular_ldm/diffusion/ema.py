from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters.

    Maintaining a slowly-updated copy of the denoiser's weights and sampling
    from that copy is standard practice in diffusion training — it markedly
    reduces sample variance and is used in essentially every strong diffusion
    model (Ho et al. 2020, Nichol & Dhariwal 2021, Rombach et al. 2022).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # Buffers like integer counters are copied verbatim.
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s parameters with the EMA weights (in place)."""
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        return deepcopy(self.shadow)
