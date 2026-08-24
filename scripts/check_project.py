from pathlib import Path
import compileall
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import generator_aware_supcon_loss, phishing_domain_alignment_loss
from src.calibration import calibrated_evaluation
from src.model import PhishGenModel
from src.preprocessing import STATIC_FEATURE_NAMES, extract_static_features


root = ROOT
syntax_ok = compileall.compile_dir(root / "src", quiet=1) and compileall.compile_dir(
    root / "scripts", quiet=1
)

features = extract_static_features(
    "<html><body><form><input type='password'><button>Login</button></form></body></html>",
    "https://secure.example.invalid/login",
)
assert len(features) == len(STATIC_FEATURE_NAMES)

model = PhishGenModel(
    use_html=False,
    use_dom=True,
    use_static=True,
    dom_hidden_size=32,
    dom_layers=1,
    dom_heads=4,
    max_dom_length=8,
    static_feature_size=len(STATIC_FEATURE_NAMES),
    fusion_hidden_size=32,
    projection_size=16,
    num_generators=3,
)
batch_size = 4
output = model(
    torch.zeros((batch_size, 1), dtype=torch.long),
    torch.ones((batch_size, 1), dtype=torch.long),
    torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]] * batch_size),
    torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]] * batch_size),
    torch.randn(batch_size, len(STATIC_FEATURE_NAMES)),
)
assert output["logits"].shape == (batch_size,)
assert output["generator_logits"].shape == (batch_size, 3)

labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
generators = torch.tensor([0, 0, 1, 2])
templates = torch.tensor([1, 2, 3, 4])
domains = torch.tensor([0, 0, 0, 1])
contrastive = generator_aware_supcon_loss(
    output["projection"], labels, generators, template_ids=templates
)
alignment = phishing_domain_alignment_loss(output["embedding"], labels, domains)
assert torch.isfinite(contrastive)
assert torch.isfinite(alignment)

calibration = calibrated_evaluation(
    [0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9],
    [0, 0, 1, 1], [0.2, 0.4, 0.8, 0.95], target_fpr=0.01,
)
assert set(calibration) == {"f1", "target_fpr", "target_fpr_confident"}
assert calibration["target_fpr"]["validation"]["fpr"] <= 0.01

print("Syntax check:", "OK" if syntax_ok else "FAILED")
print("Offline model/loss smoke test: OK")
raise SystemExit(0 if syntax_ok else 1)
