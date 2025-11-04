from dexbotic.exp.utils import require_config_keys
from dexbotic.model.memvla.action_model.action_models import ActionModel


@require_config_keys(["action_model_type", "hidden_size", "action_dim", "chunk_size"])
def build_action_model(config):
    model_type = config.action_model_type
    in_channels = config.action_dim
    token_size = config.hidden_size
    future_action_window_size = config.chunk_size - \
        1

    if hasattr(config, "per_token_size"):
        per_token_size = config.per_token_size
        use_per_attn = True
    else:
        per_token_size = None
        use_per_attn = False

    assert "DiT" in model_type
    action_model = ActionModel(
        token_size=token_size,
        model_type=model_type,
        in_channels=in_channels,
        future_action_window_size=future_action_window_size,
        use_per_attn=use_per_attn,
        per_token_size=per_token_size,
    )

    return action_model
