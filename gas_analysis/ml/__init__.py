"""
Machine Learning Enhanced Gas Sensing Module

This module provides ML-based spectral analysis for improved gas detection:
1. SpectralFeatureEngineering - First-derivative convolution for weak absorber enhancement
2. CNN1DSpectralAnalyzer - 1D-CNN for concentration prediction
3. DetectionLimitCalculator - IUPAC-compliant LoD/LoQ calculation
4. StatisticalAnalysis - Publication-grade statistical testing
5. PublicationPlots - Tier-1 journal quality visualization

Target improvements over baseline ZnO-NCF sensor:
- LoD: 3.26 ppm → <1 ppm (77% reduction)
- Sensitivity: 0.116 → 0.156 nm/ppm (35% improvement)
- Clinical accuracy: 96% for diabetes screening

Based on methodology from:
- "1-s2.0-S0925400525000607-main.pdf" - Spectral feature engineering
- "Highly sensitive and real-time detection of acetone biomarker" - ZnO-NCF sensor
"""

from .spectral_feature_engineering import (
    SpectralFeatureEngineering,
    DetectionLimitCalculator,
    engineer_features_for_gas_spectra
)

from .statistical_analysis import (
    StatisticalAnalysis,
    ClinicalDiagnosticMetrics,
    CalibrationAnalysis,
    generate_publication_statistics_report
)

from .publication_plots import (
    plot_sensitivity_comparison,
    plot_feature_engineering_demonstration,
    plot_training_curves,
    plot_roc_curve,
    plot_allan_deviation,
    plot_performance_comparison_table,
    setup_publication_style
)

# Optional CNN module (requires TensorFlow)
try:
    from .cnn_spectral_model import (
        CNN1DSpectralAnalyzer,
        SpectralDataAugmentor,
        create_training_pipeline
    )
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

__all__ = [
    # Feature Engineering
    'SpectralFeatureEngineering',
    'DetectionLimitCalculator',
    'engineer_features_for_gas_spectra',
    
    # Statistical Analysis
    'StatisticalAnalysis',
    'ClinicalDiagnosticMetrics',
    'CalibrationAnalysis',
    'generate_publication_statistics_report',
    
    # Visualization
    'plot_sensitivity_comparison',
    'plot_feature_engineering_demonstration',
    'plot_training_curves',
    'plot_roc_curve',
    'plot_allan_deviation',
    'plot_performance_comparison_table',
    'setup_publication_style',
    
    # CNN (if available)
    'CNN_AVAILABLE',
]

# Add CNN exports if available
if CNN_AVAILABLE:
    __all__.extend([
        'CNN1DSpectralAnalyzer',
        'SpectralDataAugmentor',
        'create_training_pipeline'
    ])
