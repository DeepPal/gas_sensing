"""
1D Convolutional Neural Network for Spectral Gas Detection

This module implements a 1D-CNN architecture for concentration prediction
from spectral data, with optional feature engineering preprocessing.

Architecture (based on reference paper):
    Input → Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → Conv1D(128)
    → Flatten → Dense(256) → Dense(128) → Output

Key features:
    1. Accepts raw or feature-engineered spectra
    2. Multi-output for simultaneous detection of multiple VOCs
    3. Built-in data augmentation for small datasets
    4. Uncertainty estimation via MC Dropout

Target performance:
    - MSE reduction: 10× for weak absorbers (acetone)
    - R² improvement: 0.95 → 0.98
    - Noise robustness: Works at SNR > 50 dB

Author: ML-Enhanced Gas Sensing Pipeline
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import warnings

# Check for TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. CNN models will use sklearn fallback.")

# Sklearn fallback
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class CNN1DSpectralAnalyzer:
    """
    1D Convolutional Neural Network for spectral gas detection.
    
    Processes raw or feature-engineered spectra to predict gas concentrations.
    
    Architecture:
        Conv1D(32, k=3) → MaxPool(2) → Dropout(0.2)
        → Conv1D(64, k=3) → MaxPool(2) → Dropout(0.2)
        → Conv1D(128, k=3) → MaxPool(2) → Dropout(0.2)
        → Flatten → Dense(256) → Dropout(0.3)
        → Dense(128) → Dropout(0.2) → Output(n_species)
    
    Attributes:
        input_length (int): Length of input spectral data
        n_outputs (int): Number of output species/concentrations
        model: Keras model or sklearn fallback
        history: Training history
        scaler_input: Input data scaler
        scaler_output: Output data scaler
    """
    
    def __init__(
        self,
        input_length: int,
        n_outputs: int = 1,
        use_tf: bool = True
    ):
        """
        Initialize CNN spectral analyzer.
        
        Parameters
        ----------
        input_length : int
            Length of input spectral data
        n_outputs : int
            Number of output species (regression targets)
        use_tf : bool
            Use TensorFlow if available (falls back to sklearn if not)
        """
        self.input_length = input_length
        self.n_outputs = n_outputs
        self.use_tf = use_tf and TF_AVAILABLE
        self.model = None
        self.history = None
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self._is_fitted = False
        
    def build_model(
        self,
        filters: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        dense_units: List[int] = [256, 128],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Build 1D-CNN architecture.
        
        Parameters
        ----------
        filters : list
            Number of filters for each Conv1D layer
        kernel_size : int
            Kernel size for convolutions
        dense_units : list
            Units for dense layers
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Adam optimizer learning rate
            
        Returns
        -------
        model : keras.Model or sklearn estimator
        """
        if self.use_tf:
            return self._build_keras_model(
                filters, kernel_size, dense_units, dropout_rate, learning_rate
            )
        else:
            return self._build_sklearn_model()
    
    def _build_keras_model(
        self,
        filters: List[int],
        kernel_size: int,
        dense_units: List[int],
        dropout_rate: float,
        learning_rate: float
    ):
        """Build Keras CNN model."""
        model = models.Sequential([
            layers.Input(shape=(self.input_length, 1)),
        ])
        
        # Convolutional blocks
        for i, n_filters in enumerate(filters):
            model.add(layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
        
        # Flatten and dense layers
        model.add(layers.Flatten(name='flatten'))
        
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            model.add(layers.Dropout(
                dropout_rate + 0.1 if i == 0 else dropout_rate,
                name=f'dropout_dense_{i+1}'
            ))
        
        # Output layer (regression)
        model.add(layers.Dense(
            self.n_outputs,
            activation='linear',
            name='output'
        ))
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def _build_sklearn_model(self):
        """Build sklearn ensemble as fallback."""
        # Use Gradient Boosting as a robust alternative to CNN
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        return self.model
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        normalize: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input spectra
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target concentrations
        test_size : float
            Fraction for validation set
        normalize : bool
            Whether to normalize data
        random_state : int
            Random seed for reproducibility
            
        Returns
        -------
        X_train, X_val, y_train, y_val : arrays
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize
        if normalize:
            X_train = self.scaler_input.fit_transform(X_train)
            X_val = self.scaler_input.transform(X_val)
            
            y_train = self.scaler_output.fit_transform(y_train)
            y_val = self.scaler_output.transform(y_val)
        
        # Reshape for CNN (add channel dimension)
        if self.use_tf:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        return X_train, X_val, y_train, y_val
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model.
        
        Parameters
        ----------
        X_train, y_train : arrays
            Training data
        X_val, y_val : arrays, optional
            Validation data
        epochs : int
            Maximum training epochs
        batch_size : int
            Batch size
        early_stopping_patience : int
            Patience for early stopping
        verbose : int
            Verbosity level
            
        Returns
        -------
        history : dict
            Training history
        """
        if self.model is None:
            self.build_model()
        
        if self.use_tf:
            return self._train_keras(
                X_train, y_train, X_val, y_val,
                epochs, batch_size, early_stopping_patience, verbose
            )
        else:
            return self._train_sklearn(X_train, y_train)
    
    def _train_keras(
        self,
        X_train, y_train, X_val, y_val,
        epochs, batch_size, patience, verbose
    ) -> Dict:
        """Train Keras model."""
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        )
        callback_list.append(early_stop)
        
        # Learning rate reduction
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=verbose
        )
        callback_list.append(lr_reducer)
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self._is_fitted = True
        return self.history.history
    
    def _train_sklearn(self, X_train, y_train) -> Dict:
        """Train sklearn model."""
        # Flatten for sklearn
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        self.history = {
            'loss': [train_mse],
            'r2': [train_r2]
        }
        return self.history
    
    def predict(
        self,
        X: np.ndarray,
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Predict concentrations from spectra.
        
        Parameters
        ----------
        X : np.ndarray
            Input spectra
        denormalize : bool
            Whether to inverse transform predictions
            
        Returns
        -------
        predictions : np.ndarray
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        
        X = np.asarray(X)
        
        # Normalize input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler_input.transform(X)
        
        # Reshape for CNN
        if self.use_tf:
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Denormalize
        if denormalize:
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            predictions = self.scaler_output.inverse_transform(predictions)
        
        return predictions.flatten() if predictions.shape[1] == 1 else predictions
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        X_test, y_test : arrays
            Test data
            
        Returns
        -------
        metrics : dict
            MSE, RMSE, MAE, R² for each output
        """
        predictions = self.predict(X_test, denormalize=False)
        
        # Denormalize for metrics
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        predictions = self.scaler_output.inverse_transform(predictions)
        
        y_test = np.asarray(y_test)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        y_test_orig = self.scaler_output.inverse_transform(y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, predictions, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, predictions, multioutput='raw_values')
        r2 = r2_score(y_test_orig, predictions, multioutput='raw_values')
        
        return {
            'MSE': mse.tolist() if hasattr(mse, 'tolist') else [mse],
            'RMSE': rmse.tolist() if hasattr(rmse, 'tolist') else [rmse],
            'MAE': mae.tolist() if hasattr(mae, 'tolist') else [mae],
            'R2': r2.tolist() if hasattr(r2, 'tolist') else [r2]
        }
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Parameters
        ----------
        X : np.ndarray
            Input spectra
        n_samples : int
            Number of Monte Carlo samples
            
        Returns
        -------
        mean_pred : np.ndarray
            Mean prediction
        std_pred : np.ndarray
            Standard deviation (uncertainty)
        """
        if not self.use_tf:
            # Sklearn fallback - use ensemble variance
            pred = self.predict(X)
            return pred, np.zeros_like(pred)  # No uncertainty for sklearn
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler_input.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # MC Dropout sampling
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X_scaled, training=True)  # Enable dropout
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Denormalize
        mean_pred = self.scaler_output.inverse_transform(mean_pred)
        std_pred = std_pred * self.scaler_output.scale_
        
        return mean_pred.flatten(), std_pred.flatten()
    
    def save_model(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.use_tf and self.model is not None:
            self.model.save(path / 'keras_model.h5')
        
        # Save scalers and metadata
        metadata = {
            'input_length': self.input_length,
            'n_outputs': self.n_outputs,
            'use_tf': self.use_tf,
            'is_fitted': self._is_fitted
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scalers
        np.savez(
            path / 'scalers.npz',
            input_mean=self.scaler_input.mean_,
            input_scale=self.scaler_input.scale_,
            output_mean=self.scaler_output.mean_,
            output_scale=self.scaler_output.scale_
        )
    
    def get_summary(self) -> str:
        """Get model summary."""
        if self.use_tf and self.model is not None:
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            return '\n'.join(stringlist)
        else:
            return f"Sklearn model: {type(self.model).__name__}"


class SpectralDataAugmentor:
    """
    Data augmentation for spectral data.
    
    Augmentation strategies:
    1. Gaussian noise injection
    2. Wavelength shift (simulates calibration drift)
    3. Intensity scaling (simulates concentration variations)
    4. Baseline offset
    """
    
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
    
    def add_gaussian_noise(
        self,
        spectra: np.ndarray,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian noise to spectra."""
        noise = self.rng.normal(0, noise_level, spectra.shape)
        return spectra + noise
    
    def shift_wavelength(
        self,
        spectra: np.ndarray,
        max_shift: int = 2
    ) -> np.ndarray:
        """Shift spectra along wavelength axis."""
        shift = self.rng.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return spectra
        return np.roll(spectra, shift, axis=-1)
    
    def scale_intensity(
        self,
        spectra: np.ndarray,
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """Scale intensity by random factor."""
        scale = self.rng.uniform(*scale_range)
        return spectra * scale
    
    def add_baseline_offset(
        self,
        spectra: np.ndarray,
        max_offset: float = 0.05
    ) -> np.ndarray:
        """Add random baseline offset."""
        offset = self.rng.uniform(-max_offset, max_offset)
        return spectra + offset
    
    def augment(
        self,
        spectra: np.ndarray,
        n_augmented: int = 5,
        noise_level: float = 0.01,
        max_shift: int = 2,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        max_offset: float = 0.05
    ) -> np.ndarray:
        """
        Apply full augmentation pipeline.
        
        Parameters
        ----------
        spectra : np.ndarray
            Original spectra (n_samples, n_points)
        n_augmented : int
            Number of augmented samples per original
            
        Returns
        -------
        augmented : np.ndarray
            Augmented spectra including originals
        """
        spectra = np.asarray(spectra)
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        all_spectra = [spectra]
        
        for _ in range(n_augmented):
            aug = spectra.copy()
            aug = self.add_gaussian_noise(aug, noise_level)
            aug = self.shift_wavelength(aug, max_shift)
            aug = self.scale_intensity(aug, scale_range)
            aug = self.add_baseline_offset(aug, max_offset)
            all_spectra.append(aug)
        
        return np.vstack(all_spectra)


def create_training_pipeline(
    wavelengths: np.ndarray,
    spectra_dict: Dict[float, np.ndarray],
    use_feature_engineering: bool = True,
    augment_data: bool = True,
    n_augmented: int = 5
) -> Tuple[CNN1DSpectralAnalyzer, Dict]:
    """
    Create and train a complete ML pipeline for gas sensing.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array
    spectra_dict : dict
        Dictionary mapping concentration to spectra
    use_feature_engineering : bool
        Whether to apply spectral feature engineering
    augment_data : bool
        Whether to augment training data
    n_augmented : int
        Number of augmented samples per original
        
    Returns
    -------
    model : CNN1DSpectralAnalyzer
        Trained model
    results : dict
        Training results and metrics
    """
    from .spectral_feature_engineering import SpectralFeatureEngineering
    
    # Prepare data
    concentrations = sorted(spectra_dict.keys())
    spectra = np.array([spectra_dict[c] for c in concentrations])
    y = np.array(concentrations)
    
    # Apply feature engineering if requested
    if use_feature_engineering:
        sfe = SpectralFeatureEngineering(wavelengths, spectra)
        features = sfe.full_feature_engineering_pipeline()
        X = features['normalized']
    else:
        X = spectra
    
    # Augment data if requested
    if augment_data:
        augmentor = SpectralDataAugmentor()
        X_aug = augmentor.augment(X, n_augmented)
        y_aug = np.repeat(y, n_augmented + 1)
    else:
        X_aug = X
        y_aug = y
    
    # Create and train model
    model = CNN1DSpectralAnalyzer(input_length=X_aug.shape[1], n_outputs=1)
    model.build_model()
    
    # Prepare and train
    X_train, X_val, y_train, y_val = model.prepare_data(X_aug, y_aug)
    history = model.train(X_train, y_train, X_val, y_val, verbose=0)
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    
    results = {
        'training_history': history,
        'validation_metrics': metrics,
        'feature_engineering': use_feature_engineering,
        'augmentation': augment_data,
        'n_training_samples': len(X_aug)
    }
    
    return model, results


if __name__ == '__main__':
    print("1D-CNN Spectral Analyzer Module")
    print("=" * 50)
    print(f"TensorFlow available: {TF_AVAILABLE}")
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 50
    n_wavelengths = 100
    
    wavelengths = np.linspace(675, 690, n_wavelengths)
    concentrations = np.linspace(1, 10, n_samples)
    
    # Synthetic spectra
    spectra = np.zeros((n_samples, n_wavelengths))
    for i, c in enumerate(concentrations):
        peak_pos = 682.0 + c * 0.116
        spectra[i] = 0.5 + 0.3 * np.exp(-((wavelengths - peak_pos)**2) / (2 * 1.5**2))
        spectra[i] += np.random.normal(0, 0.01, n_wavelengths)
    
    # Create and train model
    model = CNN1DSpectralAnalyzer(input_length=n_wavelengths, n_outputs=1)
    model.build_model()
    
    X_train, X_val, y_train, y_val = model.prepare_data(spectra, concentrations)
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, verbose=0)
    
    metrics = model.evaluate(X_val, y_val)
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {metrics['RMSE'][0]:.4f} ppm")
    print(f"  R²: {metrics['R2'][0]:.4f}")
    print(f"  MAE: {metrics['MAE'][0]:.4f} ppm")
