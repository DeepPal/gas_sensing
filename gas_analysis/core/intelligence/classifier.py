from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GasCNN(nn.Module):
    """
    1D CNN for Gas Classification and Concentration Regression.
    """

    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Calculate flat size dynamically
        dummy_input = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            dummy_out = self.features(dummy_input)
        self.flat_size = dummy_out.view(1, -1).size(1)

        # Classifier Head (Gas Type)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Regressor Head (Concentration)
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x shape: (batch, input_length) -> (batch, 1, input_length)
        x = x.unsqueeze(1)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)

        class_logits = self.classifier(feat)
        conc_pred = self.regressor(feat)

        return class_logits, conc_pred


class CNNGasClassifier:
    """
    Wrapper for training and inference of GasCNN.
    """

    def __init__(self, input_length: int = 1000, num_classes: int = 4, device: str = "cpu"):
        self.input_length = input_length
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = GasCNN(input_length, num_classes).to(self.device)
        self.class_map: dict[int, str] = {}
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_label: np.ndarray,
        y_conc: np.ndarray,
        class_names: list[str],
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> dict[str, list[float]]:
        self.class_map = {i: name for i, name in enumerate(class_names)}

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_label_tensor = torch.tensor(y_label, dtype=torch.long)
        y_conc_tensor = torch.tensor(y_conc, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_label_tensor, y_conc_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        history = {"loss": [], "cls_acc": []}

        self.model.train()
        for _epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels, concs in loader:
                inputs, labels, concs = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    concs.to(self.device),
                )

                optimizer.zero_grad()
                logits, conc_preds = self.model(inputs)

                loss_cls = criterion_cls(logits, labels)
                loss_reg = criterion_reg(conc_preds, concs)

                # Multi-task loss: weight classification higher initially? Equal for now.
                loss = loss_cls + loss_reg

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            history["loss"].append(epoch_loss / len(loader))
            history["cls_acc"].append(correct / total)

        self.is_fitted = True
        return history

    def predict(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Returns (predicted_gas_names, predicted_concentrations)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, conc_preds = self.model(X_tensor)
            _, predicted_ids = torch.max(logits, 1)

        gas_names = [self.class_map.get(idx.item(), "Unknown") for idx in predicted_ids]
        concentrations = conc_preds.cpu().numpy().flatten()

        return gas_names, concentrations

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "config": {
                "input_length": self.input_length,
                "num_classes": self.num_classes,
                "class_map": self.class_map,
            },
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device="cpu"):
        state = torch.load(path, map_location=device)
        cfg = state["config"]
        instance = cls(
            input_length=cfg["input_length"],
            num_classes=cfg["num_classes"],
            device=device,
        )
        instance.model.load_state_dict(state["model_state"])
        instance.class_map = {int(k): v for k, v in cfg["class_map"].items()}
        instance.is_fitted = True
        return instance
