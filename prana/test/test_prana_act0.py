import torch

from PRANA.prana.policies.lerobot_policy_prana.src.lerobot_policy_prana.modeling_prana import PranaAct0Policy
from PRANA.prana.policies.lerobot_policy_prana.src.lerobot_policy_prana.configuration_prana import PranaAct0Config


class DummyFeature:
    """Minimal feature object: only needs .shape like LeRobot FeatureSpec would."""
    def __init__(self, shape):
        self.shape = tuple(shape)


def main():
    print("Running PRANA Act0 smoke test (state-only, no images)...")

    # 1) Create your real config (ONLY fields that exist in your dataclass)
    config = PranaAct0Config(
        chunk_size=8,
        n_action_steps=1,
        dim_model=128,
        dim_feedforward=256,
        n_heads=4,
        n_encoder_layer=2,
        n_decoder_layer=2,
        n_vae_encoder_layers=2,
        latent_dim=32,
        use_vae=True,
        kl_weight=1.0,
        temporal_ensemble_coeff=0.01,
    )

    # 2) Patch in the attributes your model expects at runtime.
    #    (Your model uses these: action_feature.shape[0], robot_state_feature.shape[0], env_state_feature.shape[0])
    config.action_feature = DummyFeature((6,))
    config.robot_state_feature = DummyFeature((6,))
    config.env_state_feature = DummyFeature((5,))   # make it truthy AND usable

    # 3) Disable vision *without passing it to __init__* (since your dataclass doesn't accept it)
    config.image_features = False

    # Optional: avoid device auto-selection surprises in some lerobot versions
    # (only if your PreTrainedConfig has it; harmless if it doesn't)
    try:
        config.device = "cpu"
    except Exception:
        pass

    # 4) Validate (this should now pass because env_state_feature is truthy)
    config.validate_features()

    # 5) Instantiate policy
    policy = PranaAct0Policy(config)

    # 6) Make a batch consistent with your model + loss
    B, S, A = 2, config.chunk_size, config.action_feature.shape[0]
    batch = {
        "observation.state": torch.randn(B, config.robot_state_feature.shape[0]),
        "observation.env_state": torch.randn(B, config.env_state_feature.shape[0]),
        "action": torch.randn(B, S, A),
        "action_is_pad": torch.zeros(B, S, dtype=torch.bool),
    }

    # 7) Forward pass (train)
    policy.train()
    loss, logs = policy(batch)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert "l1_loss" in logs, f"Missing l1_loss in logs: {logs}"
    assert "kld_loss" in logs, f"Missing kld_loss in logs: {logs}"

    # 8) Inference pass (temporal ensembling path)
    policy.eval()
    policy.reset()
    with torch.no_grad():
        action = policy.select_action({
            "observation.state": torch.randn(B, config.robot_state_feature.shape[0]),
            "observation.env_state": torch.randn(B, config.env_state_feature.shape[0]),
        })

    assert action.shape == (B, A), f"Bad action shape: {action.shape} expected {(B, A)}"

    print("✅ PRANA Act0 smoke test PASSED")
    print("loss:", float(loss.item()), "logs:", logs)
    print("sample action[0]:", action[0].cpu().numpy())


if __name__ == "__main__":
    main()
