"""
Basic smoke tests for the model-building and prediction pipeline.
Run with: pytest tests/
"""

import torch
from src.model import build_model


def test_build_model_output_shape():
    num_classes = 10
    model = build_model(num_classes=num_classes, backbone="resnet50", freeze_backbone=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, num_classes)


def test_build_model_invalid_backbone_raises():
    try:
        build_model(num_classes=5, backbone="not_a_real_backbone")
        assert False, "Expected ValueError for invalid backbone"
    except ValueError:
        pass
