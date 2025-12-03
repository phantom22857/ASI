import numpy as np
from unittest.mock import MagicMock, patch
from app.predict import predict

def test_predict_returns_valid_class():
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1])

    with patch("app.predict.model", fake_model):
        fake_features = [0.5] * 30
        result = predict(fake_features)

        assert isinstance(result, (int, np.integer))
        assert result in [0, 1]
