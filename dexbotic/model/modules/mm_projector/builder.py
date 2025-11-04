import re

import torch.nn as nn

from dexbotic.exp.utils import require_config_keys


@require_config_keys(['mm_projector_type', 'mm_hidden_size', 'hidden_size'])
def build_vision_projector(config):
    """
    Build the projector module for the vision tower.
    Use the config to parse parameters.
    The config should contain the following parameters:
        - mm_projector_type: the type of the projector module
        - mm_hidden_size: the hidden size of the vision tower
        - hidden_size: the hidden size of the language model
    """
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')
