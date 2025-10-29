import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import logging
from config.config_loader import load_config

# Load configuration
config = load_config()

# Set up logging
logger = logging.getLogger(__name__)

def load_spectral_data(file_path: Union[str, Path], 
                      skiprows: int = config['data']['skiprows'], 
                      header: Optional[int] = config['data']['header'],
                      names: Optional[List[str]] = config['data']['names'],
                      **kwargs) -> pd.DataFrame:
    """
    Load a single spectral data file.
    
    Args:
        file_path: Path to the CSV file
        skiprows: Number of rows to skip
        header: Row number to use as column names
        names: List of column names to use
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        pd.DataFrame: Loaded spectral data with columns ['wavelength', 'intensity']
    """
    try:
        # Set default column names if not provided
        if names is None:
            names = ['wavelength', 'intensity']
            
        data = pd.read_csv(
            file_path, 
            skiprows=skiprows,
            header=header,
            names=names,
            **kwargs
        )
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def load_experiment_directory(base_dir: Union[str, Path], 
                            concentration: str, 
                            run: int) -> Dict[str, pd.DataFrame]:
    """
    Load all data files from a specific experiment run.
    
    Args:
        base_dir: Base directory containing the 'Mixed gas' folder
        concentration: Concentration level (e.g., '0.5 ppm')
        run: Run number (1-5)
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping timestamps to dataframes
    """
    base_dir = Path(base_dir)
    exp_dir = base_dir / f"{concentration} EtOH IPA MeOH-{run}"
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Directory not found: {exp_dir}")
        
    data_dict = {}
    
    for file in exp_dir.glob("*.csv"):
        try:
            # Extract timestamp from filename
            timestamp = file.stem.split('_')[-1]
            data = load_spectral_data(file)
            data_dict[timestamp] = data
            logger.info(f"Loaded data for timestamp: {timestamp}")
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            
    return data_dict

def load_reference_data(base_dir: Union[str, Path], gas_type: str = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load reference data files.
    
    Args:
        base_dir: Base directory containing the reference files
        gas_type: Specific gas type to load (None for all)
                  Options: 'IPA', 'MeOH', 'EtOH', 'MIX'
        
    Returns:
        If gas_type is specified: Single DataFrame with reference data
        Else: Dictionary mapping gas names to DataFrames
    """
    base_dir = Path(base_dir)
    ref_files = {
        'IPA': 'ref AuMutiMIP-IPA.csv',
        'MIX': 'ref AuMutiMIP-MIX.csv',
        'MeOH': 'ref AuMutiMIP-MeOH.csv',
        'EtOH': 'ref MutiAuMIP-EtOH.csv'
    }
    
    # If specific gas type is requested
    if gas_type is not None:
        gas_type = gas_type.upper()
        if gas_type not in ref_files:
            raise ValueError(f"Invalid gas_type: {gas_type}. Must be one of {list(ref_files.keys())}")
        
        file_path = base_dir / ref_files[gas_type]
        if not file_path.exists():
            raise FileNotFoundError(f"Reference file not found: {file_path}")
            
        try:
            df = load_spectral_data(file_path)
            logger.info(f"Successfully loaded reference data for {gas_type}")
            return df
        except Exception as e:
            logger.error(f"Error loading reference data for {gas_type}: {str(e)}")
            raise
    
    # Otherwise load all reference data
    ref_data = {}
    for gas, filename in ref_files.items():
        file_path = base_dir / filename
        if file_path.exists():
            try:
                ref_data[gas] = load_spectral_data(file_path)
                logger.info(f"Loaded reference data for {gas}")
            except Exception as e:
                logger.error(f"Error loading reference data for {gas}: {str(e)}")
        else:
            logger.warning(f"Reference file not found: {file_path}")
            
    return ref_data

def preprocess_spectra(data: pd.DataFrame, 
                      reference: Optional[pd.DataFrame] = None,
                      baseline_correction: bool = True,
                      normalize: bool = True) -> pd.DataFrame:
    """
    Preprocess spectral data.
    
    Args:
        data: Input spectral data
        reference: Reference spectrum for baseline correction
        baseline_correction: Whether to perform baseline correction
        normalize: Whether to normalize the spectrum
        
    Returns:
        pd.DataFrame: Preprocessed spectral data
    """
    processed = data.copy()
    
    # Baseline correction
    if baseline_correction and reference is not None:
        processed['intensity'] = processed['intensity'] - reference['intensity']
    
    # Normalization
    if normalize:
        min_val = processed['intensity'].min()
        max_val = processed['intensity'].max()
        if max_val > min_val:  # Avoid division by zero
            processed['intensity'] = (processed['intensity'] - min_val) / (max_val - min_val)
    
    return processed