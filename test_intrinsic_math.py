"""Small CPU sanity checks for Phase 5.3 intrinsic reward math.

Run: python test_intrinsic_math.py
Does not load checkpoints or start environments.
"""
import torch
from torch.nn import functional as F

from disagreement_ensemble import DisagreementEnsemble


def main():
    torch.manual_seed(4)
    ensemble = DisagreementEnsemble(12, 5, target_dim=8, hidden_dim=16, num_heads=3)
    ensemble.eval()
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)

    features = torch.randn(3, 6, 12)
    action = F.one_hot(torch.randint(5, (3, 6)), 5).float()
    disagreement = ensemble.disagreement(features[:, :-1], action[:, :-1])
    assert disagreement.shape == (3, 5)
    assert torch.isfinite(disagreement).all()
    assert (disagreement >= 0).all()

    beta, scale, clip = 0.05, 0.01035, 3.0
    bonus = F.pad(beta * (disagreement / scale).clamp(0, clip), (1, 0)).unsqueeze(-1)
    assert bonus.shape == (3, 6, 1)
    assert (bonus[:, 0] == 0).all(), "Bonus must apply to next-state reward, not initial reward"
    assert float(bonus.max()) <= beta * clip + 1e-6
    assert not bonus.requires_grad

    # Pure offline/bootstrap sanity: input features are detached in the auxiliary.
    loss, per_head = ensemble.bootstrap_loss(features.reshape(-1, 12), action.reshape(-1, 5),
                                             torch.randn(18, 12))
    assert loss.isfinite()
    assert per_head.shape == (3,)
    print("[PASS] Ensemble shape, finite/nonnegative disagreement, bonus alignment, cap, and bootstrap loss")
    print("[NOTE] This is a CPU math smoke test; it does not validate full Dreamer training.")


if __name__ == "__main__":
    main()
