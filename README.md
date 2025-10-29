# Gas Analysis Package

A professional Python package for gas spectral analysis with advanced processing and machine learning capabilities.

## Features

- Load and preprocess spectral data
- Advanced gas analysis using machine learning
- Visualization tools for spectra and calibration curves
- Configurable and extensible architecture

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gas_analysis.git
   cd gas_analysis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

```python
from gas_analysis import GasAnalyzer, load_spectral_data, plot_spectra

# Load and preprocess data
data = load_spectral_data('path/to/your/data.csv')

# Create analyzer instance
analyzer = GasAnalyzer()

# Perform analysis
results = analyzer.analyze(data)

# Visualize results
plot_spectra(results)
```

## Project Structure

```
gas_analysis/
├── config/             # Configuration files
├── data/               # Sample data
├── gas_analysis/       # Main package
│   ├── __init__.py
│   ├── analyzer.py     # Core analysis logic
│   ├── data_loader.py  # Data loading utilities
│   └── visualization.py# Plotting functions
├── output/             # Analysis outputs
├── scripts/            # Utility scripts
├── tests/              # Test files
├── requirements.txt    # Dependencies
└── setup.py            # Package setup file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
