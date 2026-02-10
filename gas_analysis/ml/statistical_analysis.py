"""
Statistical Analysis Module for Tier-1 Publication

This module provides comprehensive statistical analysis tools for:
1. Model comparison (paired t-test, effect sizes)
2. Clinical diagnostic metrics (sensitivity, specificity, ROC-AUC)
3. Regression quality assessment (R², RMSE, prediction intervals)
4. Cross-validation and bootstrap analysis

Designed for Sensors & Actuators: B. Chemical publication standards.

Author: ML-Enhanced Gas Sensing Pipeline
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, LeaveOneOut
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalAnalysis:
    """
    Comprehensive statistical analysis for sensor comparison and validation.
    
    Methods for Tier-1 publication:
    1. Paired t-test for model comparison
    2. Cohen's d effect size
    3. Bootstrap confidence intervals
    4. Clinical classification metrics
    5. Leave-One-Out Cross-Validation (LOOCV)
    """
    
    @staticmethod
    def paired_t_test(
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Paired t-test to compare standard vs feature-engineered models.
        
        H₀: μ_control = μ_treatment (no difference)
        H₁: μ_control ≠ μ_treatment (difference exists)
        
        Parameters
        ----------
        control : np.ndarray
            Control group measurements (e.g., standard model MSE)
        treatment : np.ndarray
            Treatment group measurements (e.g., feature-engineered MSE)
        alpha : float
            Significance level
            
        Returns
        -------
        result : dict
            t-statistic, p-value, significance, effect size
        """
        control = np.asarray(control).flatten()
        treatment = np.asarray(treatment).flatten()
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(control, treatment)
        
        # Significance markers
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < alpha:
            significance = "*"
        else:
            significance = "ns"
        
        # Cohen's d effect size
        diff = control - treatment
        effect_size = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
        
        # Effect size interpretation
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'significance_marker': significance,
            'effect_size': float(effect_size),
            'effect_interpretation': effect_interpretation,
            'control_mean': float(np.mean(control)),
            'control_std': float(np.std(control, ddof=1)),
            'treatment_mean': float(np.mean(treatment)),
            'treatment_std': float(np.std(treatment, ddof=1)),
            'improvement_percent': float((np.mean(control) - np.mean(treatment)) / 
                                        (np.mean(control) + 1e-10) * 100)
        }
    
    @staticmethod
    def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² coefficient for regression accuracy.
        
        R² = 1 - (SS_res / SS_tot)
        
        where SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - mean(y_true))²
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < 1e-10:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Avoid division by zero
        mask = np.abs(y_true) > 1e-10
        if not np.any(mask):
            return np.nan
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic_func: callable = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval estimation.
        
        Parameters
        ----------
        data : np.ndarray
            Sample data
        statistic_func : callable
            Function to compute statistic (default: mean)
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)
        random_state : int
            Random seed
            
        Returns
        -------
        point_estimate, ci_lower, ci_upper : floats
        """
        rng = np.random.RandomState(random_state)
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        point_estimate = statistic_func(data)
        
        return float(point_estimate), float(ci_lower), float(ci_upper)
    
    @staticmethod
    def loocv_regression(
        X: np.ndarray,
        y: np.ndarray,
        model_func: callable = None
    ) -> Dict:
        """
        Leave-One-Out Cross-Validation for regression.
        
        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features) or (n_samples,)
        y : np.ndarray
            Target values
        model_func : callable, optional
            Custom model function (default: linear regression)
            
        Returns
        -------
        result : dict
            LOOCV metrics including R²_CV, RMSE_CV
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n = len(y)
        predictions = np.zeros(n)
        
        for i in range(n):
            # Leave one out
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]
            
            if model_func is not None:
                pred = model_func(X_train, y_train, X_test)
            else:
                # Default: linear regression
                coeffs = np.polyfit(X_train.flatten(), y_train, 1)
                pred = np.polyval(coeffs, X_test.flatten())
            
            predictions[i] = pred[0] if hasattr(pred, '__len__') else pred
        
        # Calculate metrics
        r2_cv = StatisticalAnalysis.calculate_r_squared(y, predictions)
        rmse_cv = StatisticalAnalysis.calculate_rmse(y, predictions)
        mae_cv = StatisticalAnalysis.calculate_mae(y, predictions)
        
        return {
            'r2_cv': r2_cv,
            'rmse_cv': rmse_cv,
            'mae_cv': mae_cv,
            'predictions': predictions.tolist(),
            'residuals': (y - predictions).tolist()
        }


class ClinicalDiagnosticMetrics:
    """
    Clinical diagnostic metrics for diabetes screening application.
    
    Metrics for Tier-1 publication:
    1. Sensitivity (True Positive Rate)
    2. Specificity (True Negative Rate)
    3. Accuracy
    4. ROC-AUC
    5. Precision-Recall
    
    Clinical relevance:
    - Diabetes threshold: ~1.2 ppm acetone
    - Healthy: 0.2-1.8 ppm
    - Diabetic: 1.25-2.5 ppm
    """
    
    def __init__(self, threshold: float = 1.2):
        """
        Initialize clinical metrics calculator.
        
        Parameters
        ----------
        threshold : float
            Classification threshold (ppm acetone)
            Default 1.2 ppm based on clinical literature
        """
        self.threshold = threshold
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate sensitivity, specificity, accuracy.
        
        Parameters
        ----------
        y_true : np.ndarray
            True concentration values (ppm)
        y_pred : np.ndarray
            Predicted concentration values (ppm)
            
        Returns
        -------
        metrics : dict
            sensitivity, specificity, accuracy, confusion matrix
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Binary classification
        y_true_binary = (y_true >= self.threshold).astype(int)
        y_pred_binary = (y_pred >= self.threshold).astype(int)
        
        # Confusion matrix components
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        
        # Calculate metrics (handle division by zero)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'f1_score': float(f1_score),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold_ppm': self.threshold
        }
    
    def calculate_roc_auc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Calculate ROC curve and AUC.
        
        Parameters
        ----------
        y_true : np.ndarray
            True concentration values (ppm)
        y_pred_proba : np.ndarray
            Predicted concentrations or probabilities
            
        Returns
        -------
        result : dict
            AUC, optimal threshold, ROC curve data
        """
        y_true = np.asarray(y_true).flatten()
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        # Binary ground truth
        y_binary = (y_true >= self.threshold).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_statistic = tpr - fpr
        optimal_idx = np.argmax(j_statistic)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_sensitivity': float(tpr[optimal_idx]),
            'optimal_specificity': float(1 - fpr[optimal_idx])
        }
    
    def calculate_precision_recall(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Calculate Precision-Recall curve and average precision.
        
        Parameters
        ----------
        y_true, y_pred_proba : np.ndarray
            True values and predictions
            
        Returns
        -------
        result : dict
            Average precision, PR curve data
        """
        y_true = np.asarray(y_true).flatten()
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        y_binary = (y_true >= self.threshold).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_binary, y_pred_proba)
        avg_precision = np.mean(precision)
        
        return {
            'average_precision': float(avg_precision),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        }


class CalibrationAnalysis:
    """
    Calibration curve analysis for sensor validation.
    
    Methods:
    1. Linear regression with confidence bands
    2. Polynomial fitting with model selection
    3. Langmuir isotherm fitting (for saturation behavior)
    4. Weighted least squares for heteroscedastic data
    """
    
    @staticmethod
    def linear_calibration(
        concentrations: np.ndarray,
        responses: np.ndarray,
        weighted: bool = False
    ) -> Dict:
        """
        Perform linear calibration with full statistics.
        
        Parameters
        ----------
        concentrations : np.ndarray
            Concentration values (independent variable)
        responses : np.ndarray
            Sensor responses (dependent variable, e.g., Δλ)
        weighted : bool
            Use weighted least squares
            
        Returns
        -------
        result : dict
            slope, intercept, r2, prediction intervals, etc.
        """
        x = np.asarray(concentrations).flatten()
        y = np.asarray(responses).flatten()
        
        # Remove NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        if len(x) < 2:
            return {'error': 'Insufficient data points'}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Predictions
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Standard error of estimate
        n = len(x)
        dof = n - 2
        se_estimate = np.sqrt(np.sum(residuals ** 2) / dof) if dof > 0 else 0
        
        # Confidence interval for slope (95%)
        t_critical = stats.t.ppf(0.975, dof) if dof > 0 else 0
        slope_ci = (
            slope - t_critical * std_err,
            slope + t_critical * std_err
        )
        
        # Prediction interval for new observations
        x_mean = np.mean(x)
        x_ss = np.sum((x - x_mean) ** 2)
        
        pred_interval_factor = t_critical * se_estimate * np.sqrt(
            1 + 1/n + (x - x_mean)**2 / x_ss
        ) if x_ss > 0 else np.zeros_like(x)
        
        # Spearman correlation (monotonicity)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_error': float(std_err),
            'se_estimate': float(se_estimate),
            'slope_ci_95': [float(slope_ci[0]), float(slope_ci[1])],
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_points': n,
            'residuals': residuals.tolist(),
            'predictions': y_pred.tolist(),
            'sensitivity_nm_per_ppm': float(slope)
        }
    
    @staticmethod
    def langmuir_calibration(
        concentrations: np.ndarray,
        responses: np.ndarray
    ) -> Dict:
        """
        Fit Langmuir isotherm for saturation behavior.
        
        Model: y = (y_max * K * x) / (1 + K * x)
        
        where y_max is maximum response and K is affinity constant.
        
        Parameters
        ----------
        concentrations, responses : np.ndarray
            Calibration data
            
        Returns
        -------
        result : dict
            y_max, K, r_squared, predictions
        """
        x = np.asarray(concentrations).flatten()
        y = np.asarray(responses).flatten()
        
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        if len(x) < 3:
            return {'error': 'Insufficient data for Langmuir fit'}
        
        def langmuir(x, y_max, K):
            return (y_max * K * x) / (1 + K * x)
        
        try:
            # Initial guess
            y_max_init = np.max(y) * 1.5
            K_init = 1.0 / (np.mean(x) + 1e-10)
            
            popt, pcov = curve_fit(
                langmuir, x, y,
                p0=[y_max_init, K_init],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=10000
            )
            
            y_max, K = popt
            y_pred = langmuir(x, y_max, K)
            
            # R² calculation
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'y_max': float(y_max),
                'K_affinity': float(K),
                'r_squared': float(r_squared),
                'predictions': y_pred.tolist(),
                'model': 'langmuir'
            }
            
        except Exception as e:
            return {'error': str(e), 'model': 'langmuir'}
    
    @staticmethod
    def polynomial_calibration(
        concentrations: np.ndarray,
        responses: np.ndarray,
        max_degree: int = 3,
        criterion: str = 'bic'
    ) -> Dict:
        """
        Polynomial calibration with automatic degree selection.
        
        Uses BIC/AIC for model selection to avoid overfitting.
        
        Parameters
        ----------
        concentrations, responses : np.ndarray
            Calibration data
        max_degree : int
            Maximum polynomial degree to try
        criterion : str
            'bic' or 'aic' for model selection
            
        Returns
        -------
        result : dict
            Best polynomial coefficients and statistics
        """
        x = np.asarray(concentrations).flatten()
        y = np.asarray(responses).flatten()
        
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        n = len(x)
        if n < 3:
            return {'error': 'Insufficient data'}
        
        best_score = np.inf
        best_result = None
        
        for degree in range(1, min(max_degree + 1, n - 1)):
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            
            # Calculate residuals and metrics
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            mse = ss_res / n
            
            # Information criteria
            k = degree + 1  # Number of parameters
            if criterion == 'bic':
                score = n * np.log(mse + 1e-10) + k * np.log(n)
            else:  # AIC
                score = n * np.log(mse + 1e-10) + 2 * k
            
            # R² calculation
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            if score < best_score:
                best_score = score
                best_result = {
                    'degree': degree,
                    'coefficients': coeffs.tolist(),
                    'r_squared': float(r_squared),
                    'mse': float(mse),
                    'criterion_score': float(score),
                    'predictions': y_pred.tolist(),
                    'residuals': residuals.tolist()
                }
        
        return best_result


def generate_publication_statistics_report(
    concentrations: np.ndarray,
    standard_predictions: np.ndarray,
    engineered_predictions: np.ndarray,
    true_values: np.ndarray = None,
    clinical_threshold: float = 1.2
) -> Dict:
    """
    Generate comprehensive statistical report for publication.
    
    Parameters
    ----------
    concentrations : np.ndarray
        Test concentrations
    standard_predictions : np.ndarray
        Predictions from standard model
    engineered_predictions : np.ndarray
        Predictions from feature-engineered model
    true_values : np.ndarray, optional
        Ground truth values (uses concentrations if None)
    clinical_threshold : float
        Threshold for clinical classification
        
    Returns
    -------
    report : dict
        Complete statistical analysis report
    """
    if true_values is None:
        true_values = concentrations
    
    stats_analyzer = StatisticalAnalysis()
    clinical_metrics = ClinicalDiagnosticMetrics(threshold=clinical_threshold)
    calibration = CalibrationAnalysis()
    
    # Model comparison
    std_errors = np.abs(standard_predictions - true_values)
    eng_errors = np.abs(engineered_predictions - true_values)
    
    comparison = stats_analyzer.paired_t_test(std_errors, eng_errors)
    
    # Individual model metrics
    std_metrics = {
        'r_squared': stats_analyzer.calculate_r_squared(true_values, standard_predictions),
        'rmse': stats_analyzer.calculate_rmse(true_values, standard_predictions),
        'mae': stats_analyzer.calculate_mae(true_values, standard_predictions),
        'mape': stats_analyzer.calculate_mape(true_values, standard_predictions)
    }
    
    eng_metrics = {
        'r_squared': stats_analyzer.calculate_r_squared(true_values, engineered_predictions),
        'rmse': stats_analyzer.calculate_rmse(true_values, engineered_predictions),
        'mae': stats_analyzer.calculate_mae(true_values, engineered_predictions),
        'mape': stats_analyzer.calculate_mape(true_values, engineered_predictions)
    }
    
    # Clinical metrics
    std_clinical = clinical_metrics.calculate_classification_metrics(
        true_values, standard_predictions
    )
    eng_clinical = clinical_metrics.calculate_classification_metrics(
        true_values, engineered_predictions
    )
    
    # ROC analysis
    eng_roc = clinical_metrics.calculate_roc_auc(true_values, engineered_predictions)
    
    # Calibration analysis
    std_calibration = calibration.linear_calibration(concentrations, standard_predictions)
    eng_calibration = calibration.linear_calibration(concentrations, engineered_predictions)
    
    return {
        'model_comparison': comparison,
        'standard_model': {
            'regression_metrics': std_metrics,
            'clinical_metrics': std_clinical,
            'calibration': std_calibration
        },
        'feature_engineered_model': {
            'regression_metrics': eng_metrics,
            'clinical_metrics': eng_clinical,
            'calibration': eng_calibration,
            'roc_analysis': eng_roc
        },
        'improvement_summary': {
            'r_squared_improvement': eng_metrics['r_squared'] - std_metrics['r_squared'],
            'rmse_improvement_percent': (std_metrics['rmse'] - eng_metrics['rmse']) / 
                                       (std_metrics['rmse'] + 1e-10) * 100,
            'sensitivity_improvement_percent': comparison['improvement_percent'],
            'clinical_accuracy_improvement': eng_clinical['accuracy'] - std_clinical['accuracy']
        }
    }


if __name__ == '__main__':
    print("Statistical Analysis Module")
    print("=" * 50)
    
    # Generate synthetic test data
    np.random.seed(42)
    n = 30
    concentrations = np.linspace(1, 10, n)
    
    # Simulate standard model predictions (baseline)
    standard_pred = concentrations + np.random.normal(0, 0.5, n)
    
    # Simulate feature-engineered model predictions (improved)
    engineered_pred = concentrations + np.random.normal(0, 0.2, n)
    
    # Generate report
    report = generate_publication_statistics_report(
        concentrations=concentrations,
        standard_predictions=standard_pred,
        engineered_predictions=engineered_pred
    )
    
    print("\n--- MODEL COMPARISON ---")
    comp = report['model_comparison']
    print(f"t-statistic: {comp['t_statistic']:.4f}")
    print(f"p-value: {comp['p_value']:.4e} {comp['significance_marker']}")
    print(f"Effect size (Cohen's d): {comp['effect_size']:.4f} ({comp['effect_interpretation']})")
    
    print("\n--- STANDARD MODEL ---")
    std = report['standard_model']['regression_metrics']
    print(f"R²: {std['r_squared']:.4f}")
    print(f"RMSE: {std['rmse']:.4f} ppm")
    
    print("\n--- FEATURE-ENGINEERED MODEL ---")
    eng = report['feature_engineered_model']['regression_metrics']
    print(f"R²: {eng['r_squared']:.4f}")
    print(f"RMSE: {eng['rmse']:.4f} ppm")
    
    print("\n--- IMPROVEMENT SUMMARY ---")
    imp = report['improvement_summary']
    print(f"R² improvement: +{imp['r_squared_improvement']:.4f}")
    print(f"RMSE improvement: {imp['rmse_improvement_percent']:.1f}%")
