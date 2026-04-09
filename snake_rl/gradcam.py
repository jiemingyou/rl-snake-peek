"""Grad-CAM for a DQN model.

Computes a class-discriminative saliency heatmap by weighting the activations of
a target convolutional layer by the gradient of the chosen action's Q-value with
respect to those activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def compute_gradcam(
    model: nn.Module,
    state: np.ndarray,
    action_index: int,
    target_layer_name: str = "conv3",
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Return a 2-D heatmap (H, W) in [0, 1] for the given action.

    Parameters
    ----------
    model : nn.Module
        The DQN (must have a layer named *target_layer_name*).
    state : np.ndarray
        Single-sample state array of shape (C, H, W).
    action_index : int
        Index of the action whose Q-value drives the gradient.
    target_layer_name : str
        Name of the Conv2d layer to extract activations/gradients from.
    device : torch.device | str
        Device to run the forward/backward pass on.

    Returns
    -------
    np.ndarray
        Heatmap of shape (H, W) with values in [0, 1].
    """
    model.eval()
    target_layer: nn.Module = getattr(model, target_layer_name)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    fwd_handle = target_layer.register_forward_hook(
        lambda _mod, _inp, out: activations.append(out)
    )
    bwd_handle = target_layer.register_full_backward_hook(
        lambda _mod, _grad_in, grad_out: gradients.append(grad_out[0])
    )

    try:
        inp = torch.from_numpy(state).unsqueeze(0).to(device).requires_grad_(False)
        # We need gradients w.r.t. the activations, not the input
        q_values = model(inp)
        score = q_values[0, action_index]

        model.zero_grad()
        score.backward()

        act = activations[0].detach()   # (1, C, H, W)
        grad = gradients[0].detach()    # (1, C, H, W)

        # Channel-wise global-average-pooled gradient -> weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze()                 # (H, W)

        cam_np: np.ndarray = cam.cpu().numpy()
        cam_max = cam_np.max()
        if cam_max > 0:
            cam_np = cam_np / cam_max
        return cam_np

    finally:
        fwd_handle.remove()
        bwd_handle.remove()
