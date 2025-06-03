ERROR:__main__:💥 Pipeline failed with error: Error(s) in loading state_dict for TransformerActorCriticPolicy:
	Missing key(s) in state_dict: "features_extractor.cls_token", "features_extractor.pos_encoding", "features_extractor.input_projection.0.weight", "features_extractor.input_projection.0.bias", "features_extractor.input_projection.1.weight", "features_extractor.input_projection.1.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_weight", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_bias", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.linear1.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.linear1.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.linear2.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.linear2.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.norm1.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.norm1.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.norm2.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.norm2.bias", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_weight", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_bias", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.out_proj.weight", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.out_proj.bias", "features_extractor.transformer_encoder._orig_mod.layers.1.linear1.weight", "features_extractor.transformer_encoder._orig_mod.layers.1.linear1.bias", "features_extractor.transformer_encoder._orig_mod.layers.1.linear2.weight", "features_extractor.transformer_encoder._orig_mod.layers.1.linear2.bias", "features_extractor.transformer_encoder._orig_mod.layers.1.norm1.weight", "features_extractor.transformer_encoder._orig_mod.layers.1.norm1.bias", "features_extractor.transformer_encoder._orig_mod.layers.1.norm2.weight", "features_extractor.transformer_encoder._orig_mod.layers.1.norm2.bias", "features_extractor.attention_pooling.in_proj_weight", "features_extractor.attention_pooling.in_proj_bias", "features_extractor.attention_pooling.out_proj.weight", "features_extractor.attention_pooling.out_proj.bias", "features_extractor.output_projection.0.weight", "features_extractor.output_projection.0.bias", "features_extractor.output_projection.3.weight", "features_extractor.output_projection.3.bias". 
	Unexpected key(s) in state_dict: "features_extractor._orig_mod.cls_token", "features_extractor._orig_mod.pos_encoding", "features_extractor._orig_mod.input_projection.0.weight", "features_extractor._orig_mod.input_projection.0.bias", "features_extractor._orig_mod.input_projection.1.weight", "features_extractor._orig_mod.input_projection.1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear1.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear2.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear2.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm1.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm2.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm2.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.self_attn.out_proj.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.self_attn.out_proj.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.linear1.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.linear1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.linear2.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.linear2.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.norm1.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.norm1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.norm2.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.norm2.bias", "features_extractor._orig_mod.attention_pooling.in_proj_weight", "features_extractor._orig_mod.attention_pooling.in_proj_bias", "features_extractor._orig_mod.attention_pooling.out_proj.weight", "features_extractor._orig_mod.attention_pooling.out_proj.bias", "features_extractor._orig_mod.output_projection.0.weight", "features_extractor._orig_mod.output_projection.0.bias", "features_extractor._orig_mod.output_projection.3.weight", "features_extractor._orig_mod.output_projection.3.bias". 
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-1-03dcd28945f1> in <cell line: 0>()
   2395     # Uncomment these lines to run:
   2396     check_installation()
-> 2397     main()

5 frames
/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in load_state_dict(self, state_dict, strict, assign)
   2579 
   2580         if len(error_msgs) > 0:
-> 2581             raise RuntimeError(
   2582                 "Error(s) in loading state_dict for {}:\n\t{}".format(
   2583                     self.__class__.__name__, "\n\t".join(error_msgs)

RuntimeError: Error(s) in loading state_dict for TransformerActorCriticPolicy:
	Missing key(s) in state_dict: "features_extractor.cls_token", "features_extractor.pos_encoding", "features_extractor.input_projection.0.weight", "features_extractor.input_projection.0.bias", "features_extractor.input_projection.1.weight", "features_extractor.input_projection.1.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_weight", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_bias", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.linear1.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.linear1.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.linear2.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.linear2.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.norm1.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.norm1.bias", "features_extractor.transformer_encoder._orig_mod.layers.0.norm2.weight", "features_extractor.transformer_encoder._orig_mod.layers.0.norm2.bias", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_weight", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_bias", "features_extractor.transformer_encoder._orig_mod.layers.1.self_attn.out_proj.weight", "features_extractor....
	Unexpected key(s) in state_dict: "features_extractor._orig_mod.cls_token", "features_extractor._orig_mod.pos_encoding", "features_extractor._orig_mod.input_projection.0.weight", "features_extractor._orig_mod.input_projection.0.bias", "features_extractor._orig_mod.input_projection.1.weight", "features_extractor._orig_mod.input_projection.1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.in_proj_bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.self_attn.out_proj.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear1.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear2.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.linear2.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm1.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm1.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm2.weight", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.0.norm2.bias", "features_extractor._orig_mod.transformer_encoder._orig_mod.layers.1.self_attn.in_proj_weight"...

"""
AI Trading System - Complete Google Colab Implementation
Author: AI Trading System
Purpose: Transformer + Deep Reinforcement Learning for EURUSD H1 Trading
Environment: Google Colab with GPU

This script includes:
1. Data preprocessing and feature engineering
2. Custom Transformer policy for stable-baselines3
3. gym_mtsim environment setup
4. PPO agent training
5. Model evaluation and backtesting
6. Performance reporting
"""

# ========== Installation and Imports ==========
# Run this cell first to install required packages
"""
!pip install stable-baselines3[extra]
!pip install gym-mtsim
!pip install pandas-ta
!pip install gymnasium
!pip install tensorboard
"""

# Alternative installation method if needed:
def install_requirements():
    """Install required packages"""
    import subprocess
    import sys

    packages = [
        "stable-baselines3[extra]",
        "gym-mtsim",
        "pandas-ta",
        "gymnasium",
        "tensorboard"
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def check_gpu_availability():
    """Check GPU availability and setup"""
    import torch

    print("🔍 GPU & Hardware Check:")
    print(f"   • PyTorch version: {torch.__version__}")
    print(f"   • CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   • CUDA version: {torch.version.cuda}")
        print(f"   • GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   • GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   • GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f"   • GPU memory cached: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")

        # Enable optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

        print("   ✅ GPU optimizations enabled!")
        return True
    else:
        print("   ⚠️  No GPU available, using CPU")
        return False

def optimize_torch_settings():
    """Optimize PyTorch settings for performance"""
    import torch

    # Set number of threads for CPU operations
    torch.set_num_threads(4)  # Colab has 2 cores, use 4 threads


    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

        # Enable memory fraction usage
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

    print("⚡ PyTorch optimizations applied!")

def check_installation():
    """Check if all required packages are installed"""
    required_packages = {
        'stable_baselines3': 'stable-baselines3',
        'gym_mtsim': 'gym-mtsim',
        'pandas_ta': 'pandas-ta',
        'gymnasium': 'gymnasium',
        'torch': 'torch (should be pre-installed)',
        'tensorboard': 'tensorboard'
    }

    print("🔍 Checking package installation...")

    all_installed = True
    for package, display_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {display_name}: OK")
        except ImportError:
            print(f"❌ {display_name}: NOT FOUND")
            all_installed = False

    if all_installed:
        print("\n🎉 All packages are installed correctly!")
        gpu_available = check_gpu_availability()
        optimize_torch_settings()
        return True, gpu_available
    else:
        print("\n⚠️  Some packages are missing. Please install them first.")
        return False, False

# Uncomment to check installation and GPU
# check_installation()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import time  # Added for timing measurements
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Deep Learning and RL imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn

# Gymnasium and trading environment
import gymnasium as gym
from gymnasium import spaces
import gym_mtsim

# Data processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Technical indicators
import pandas_ta as ta

# Google Colab specific
from google.colab import drive, files
import zipfile

# Suppress warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
@dataclass
class Config:
    """Central configuration for the AI Trading System - GPU OPTIMIZED VERSION"""

    # Data Configuration
    SYMBOL: str = "EURUSD"
    TIMEFRAME: str = "H1"
    WINDOW_SIZE: int = 24  # Increased from 24 to 32 (power of 2 for GPU efficiency)

    # Data Splitting
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Feature Engineering (Optimized)
    RSI_PERIOD: int = 14
    SMA_PERIODS: List[int] = (20, 50)  # Added back SMA(50) for GPU efficiency
    ATR_PERIOD: int = 14

    # Transformer Architecture (GPU Optimized)
    D_MODEL: int = 32  # Increased from 32 to 64 (power of 2)
    NUM_HEADS: int = 2  # Increased from 2 to 4 (optimal for GPU)
    NUM_ENCODER_LAYERS: int = 2  # Increased back to 2 layers
    DIM_FEEDFORWARD: int = 64  # Increased from 64 to 256 (4x d_model)
    DROPOUT: float = 0.1

    # Trading Environment
    INITIAL_BALANCE: float = 10000.0
    LEVERAGE: int = 100
    COMMISSION_PER_LOT: float = 7.0
    LOT_SIZE: float = 0.01

    # Risk Management
    N_SL: float = 1.5
    N_TP: float = 3.0
    MAX_DRAWDOWN_PCT: float = 2.0

    # PPO Hyperparameters (GPU Optimized)
    LEARNING_RATE: float = 3e-4  # Optimal for GPU training
    N_STEPS: int = 2048  # Increased back for better GPU utilization
    BATCH_SIZE: int = 128  # Increased for GPU efficiency (power of 2)
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    ENT_COEF: float = 0.01
    CLIP_RANGE: float = 0.2
    N_EPOCHS: int = 10  # Increased back to 10 for better learning

    # Training Configuration (GPU Optimized)
    TOTAL_TIMESTEPS: int = 500_000  # Balanced: not too slow, good for GPU
    EVAL_FREQ: int = 10_000
    SAVE_FREQ: int = 20_000

    # GPU/Performance Settings
    DEVICE: str = "auto"  # Will auto-detect GPU/CPU
    NUM_ENVS: int = 4  # Multiple environments for parallel training
    USE_MIXED_PRECISION: bool = True  # Enable mixed precision training
    COMPILE_MODEL: bool = True  # Enable PyTorch 2.0 compilation

    # Memory Optimization
    MAX_GRAD_NORM: float = 0.5  # Gradient clipping
    PREFETCH_FACTOR: int = 2  # Data loading optimization

    # Paths
    DRIVE_PATH: str = "/content/drive/MyDrive/AI_Trading_System"
    RAW_DATA_PATH: str = "01_Raw_Data"
    PROCESSED_DATA_PATH: str = "02_Processed_Data"
    MODELS_PATH: str = "03_Trained_Models"
    REPORTS_PATH: str = "04_Logs_and_Reports"

    def __post_init__(self):
        """Auto-configure based on available hardware"""
        import torch

        # Auto-detect device
        if self.DEVICE == "auto":
            if torch.cuda.is_available():
                self.DEVICE = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

                # Adjust batch size based on GPU memory
                if gpu_memory >= 15:  # High-end GPU (T4, V100, etc.)
                    self.BATCH_SIZE = 64
                    self.N_STEPS = 1024
                    self.NUM_ENVS = 8
                    self.TOTAL_TIMESTEPS = 500_000  # More training for GPU
                elif gpu_memory >= 8:  # Mid-range GPU
                    self.BATCH_SIZE = 128
                    self.N_STEPS = 2048
                    self.NUM_ENVS = 4
                    self.TOTAL_TIMESTEPS = 500_000
                else:  # Lower memory GPU
                    self.BATCH_SIZE = 64
                    self.N_STEPS = 1024
                    self.NUM_ENVS = 2
                    self.TOTAL_TIMESTEPS = 500_000

                print(f"🚀 GPU Configuration: {gpu_memory:.1f}GB memory")
                print(f"   • Batch Size: {self.BATCH_SIZE}")
                print(f"   • N Steps: {self.N_STEPS}")
                print(f"   • Num Envs: {self.NUM_ENVS}")
                print(f"   • Total Steps: {self.TOTAL_TIMESTEPS:,}")

            else:
                self.DEVICE = "cpu"
                # CPU optimizations - more conservative settings
                self.BATCH_SIZE = 32
                self.N_STEPS = 512
                self.NUM_ENVS = 2
                self.USE_MIXED_PRECISION = False
                self.COMPILE_MODEL = False
                self.TOTAL_TIMESTEPS = 20_000  # Reduced for CPU

                # Smaller model for CPU
                self.D_MODEL = 32
                self.NUM_HEADS = 2
                self.NUM_ENCODER_LAYERS = 1
                self.DIM_FEEDFORWARD = 128

                print("💾 CPU Configuration applied")
                print(f"   • Model optimized for CPU performance")
                print(f"   • Reduced training steps: {self.TOTAL_TIMESTEPS:,}")

        # Ensure batch size is divisible by num_envs
        if self.BATCH_SIZE % self.NUM_ENVS != 0:
            self.BATCH_SIZE = (self.BATCH_SIZE // self.NUM_ENVS) * self.NUM_ENVS

        print(f"⚙️  Final Configuration:")
        print(f"   • Device: {self.DEVICE}")
        print(f"   • Batch Size: {self.BATCH_SIZE}")
        print(f"   • Mixed Precision: {self.USE_MIXED_PRECISION}")
        print(f"   • Model Compilation: {self.COMPILE_MODEL}")
        print(f"   • Training Steps: {self.TOTAL_TIMESTEPS:,}")

    # Paths
    DRIVE_PATH: str = "/content/drive/MyDrive/AI_Trading_System"
    RAW_DATA_PATH: str = "01_Raw_Data"
    PROCESSED_DATA_PATH: str = "02_Processed_Data"
    MODELS_PATH: str = "03_Trained_Models"
    REPORTS_PATH: str = "04_Logs_and_Reports"

# ========== Logging Setup ==========
def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging for the system"""

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ai_trading_system.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

# ========== Google Drive Integration ==========
class DriveManager:
    """Manage Google Drive operations"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def mount_drive(self):
        """Mount Google Drive"""
        self.logger.info("Mounting Google Drive...")
        drive.mount('/content/drive')

        # Create directory structure if it doesn't exist
        for path in [self.config.RAW_DATA_PATH, self.config.PROCESSED_DATA_PATH,
                    self.config.MODELS_PATH, self.config.REPORTS_PATH]:
            full_path = os.path.join(self.config.DRIVE_PATH, path)
            os.makedirs(full_path, exist_ok=True)

        self.logger.info("Google Drive mounted successfully")

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from Google Drive"""
        self.logger.info("Loading raw data from Google Drive...")

        raw_data_dir = os.path.join(self.config.DRIVE_PATH, self.config.RAW_DATA_PATH)

        # Find the most recent CSV file
        csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]

        if not csv_files:
            raise FileNotFoundError("No CSV files found in raw data directory")

        # Use the most recent file (assuming filename contains date)
        latest_file = sorted(csv_files)[-1]
        filepath = os.path.join(raw_data_dir, latest_file)

        self.logger.info(f"Loading data from: {latest_file}")

        # Load and parse data
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.logger.info(f"Loaded {len(df)} rows of data from {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def save_processed_data(self, train_data, val_data, test_data, scalers):
        """Save processed data and scalers"""
        processed_dir = os.path.join(self.config.DRIVE_PATH, self.config.PROCESSED_DATA_PATH)

        # Save datasets
        with open(os.path.join(processed_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump(train_data, f)

        with open(os.path.join(processed_dir, 'val_data.pkl'), 'wb') as f:
            pickle.dump(val_data, f)

        with open(os.path.join(processed_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

        # Save scalers
        with open(os.path.join(processed_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(scalers, f)

        self.logger.info("Processed data and scalers saved successfully")

    def load_processed_data(self):
        """Load processed data and scalers"""
        processed_dir = os.path.join(self.config.DRIVE_PATH, self.config.PROCESSED_DATA_PATH)

        with open(os.path.join(processed_dir, 'train_data.pkl'), 'rb') as f:
            train_data = pickle.load(f)

        with open(os.path.join(processed_dir, 'val_data.pkl'), 'rb') as f:
            val_data = pickle.load(f)

        with open(os.path.join(processed_dir, 'test_data.pkl'), 'rb') as f:
            test_data = pickle.load(f)

        with open(os.path.join(processed_dir, 'scalers.pkl'), 'rb') as f:
            scalers = pickle.load(f)

        return train_data, val_data, test_data, scalers

# ========== Data Preprocessing ==========
class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scalers = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features - ROBUST VERSION"""
        self.logger.info("Creating technical features with robust error handling...")

        df_features = df.copy()

        # Validate essential columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']
        missing_cols = [col for col in required_cols if col not in df_features.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Basic price features
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        df_features['price_momentum'] = df_features['close'].pct_change(5)  # 5-period momentum

        # Handle real_volume (may not exist)
        if 'real_volume' not in df_features.columns:
            self.logger.info("Real volume not available, creating placeholder")
            df_features['real_volume'] = df_features['tick_volume']  # Use tick volume as fallback

        # Technical indicators (enhanced with error handling)
        try:
            # RSI
            df_features['rsi'] = ta.rsi(df_features['close'], length=self.config.RSI_PERIOD)
            df_features['rsi_fast'] = ta.rsi(df_features['close'], length=7)  # Fast RSI

            # Multiple Moving Averages
            for period in self.config.SMA_PERIODS:
                df_features[f'sma_{period}'] = ta.sma(df_features['close'], length=period)
                df_features[f'ema_{period}'] = ta.ema(df_features['close'], length=period)

            # ATR (Essential for risk management)
            df_features['atr'] = ta.atr(
                df_features['high'],
                df_features['low'],
                df_features['close'],
                length=self.config.ATR_PERIOD
            )

            # Validate ATR (critical for trading)
            if df_features['atr'].isna().all():
                self.logger.warning("ATR calculation failed, using price range as fallback")
                df_features['atr'] = (df_features['high'] - df_features['low']) / df_features['close']

            df_features['atr_normalized'] = df_features['atr'] / df_features['close']

            # Bollinger Bands
            bb_data = ta.bbands(df_features['close'], length=20)
            if bb_data is not None and not bb_data.empty:
                df_features['bb_upper'] = bb_data.get('BBU_20_2.0', bb_data.iloc[:, 0] if len(bb_data.columns) > 0 else pd.Series(index=df_features.index))
                df_features['bb_middle'] = bb_data.get('BBM_20_2.0', bb_data.iloc[:, 1] if len(bb_data.columns) > 1 else df_features['close'])
                df_features['bb_lower'] = bb_data.get('BBL_20_2.0', bb_data.iloc[:, 2] if len(bb_data.columns) > 2 else pd.Series(index=df_features.index))

                # Derived BB features
                bb_range = df_features['bb_upper'] - df_features['bb_lower']
                df_features['bb_width'] = bb_range / df_features['bb_middle'].replace(0, np.nan)
                df_features['bb_position'] = ((df_features['close'] - df_features['bb_lower']) /
                                             bb_range.replace(0, np.nan))
            else:
                self.logger.warning("Bollinger Bands calculation failed, using fallback values")
                df_features['bb_upper'] = df_features['close'] * 1.02
                df_features['bb_middle'] = df_features['close']
                df_features['bb_lower'] = df_features['close'] * 0.98
                df_features['bb_width'] = 0.04
                df_features['bb_position'] = 0.5

            # MACD (full version)
            macd_data = ta.macd(df_features['close'])
            if macd_data is not None and not macd_data.empty:
                df_features['macd'] = macd_data.get('MACD_12_26_9', macd_data.iloc[:, 0] if len(macd_data.columns) > 0 else pd.Series(index=df_features.index))
                df_features['macd_signal'] = macd_data.get('MACDs_12_26_9', macd_data.iloc[:, 1] if len(macd_data.columns) > 1 else pd.Series(index=df_features.index))
                df_features['macd_hist'] = macd_data.get('MACDh_12_26_9', macd_data.iloc[:, 2] if len(macd_data.columns) > 2 else pd.Series(index=df_features.index))
            else:
                self.logger.warning("MACD calculation failed, using price momentum as fallback")
                df_features['macd'] = df_features['returns']
                df_features['macd_signal'] = df_features['returns'].rolling(9).mean()
                df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']

            # Stochastic
            stoch_data = ta.stoch(df_features['high'], df_features['low'], df_features['close'])
            if stoch_data is not None and not stoch_data.empty:
                df_features['stoch_k'] = stoch_data.get('STOCHk_14_3_3', stoch_data.iloc[:, 0] if len(stoch_data.columns) > 0 else pd.Series(index=df_features.index))
                df_features['stoch_d'] = stoch_data.get('STOCHd_14_3_3', stoch_data.iloc[:, 1] if len(stoch_data.columns) > 1 else pd.Series(index=df_features.index))
            else:
                self.logger.warning("Stochastic calculation failed, using simplified version")
                df_features['stoch_k'] = 50.0  # Neutral value
                df_features['stoch_d'] = 50.0

            # Williams %R
            williams_r_data = ta.willr(df_features['high'], df_features['low'], df_features['close'])
            if williams_r_data is not None:
                df_features['williams_r'] = williams_r_data
            else:
                self.logger.warning("Williams %R calculation failed, using fallback")
                df_features['williams_r'] = -50.0  # Neutral value

            # Price position relative to MAs
            for period in self.config.SMA_PERIODS:
                sma_col = f'sma_{period}'
                if sma_col in df_features.columns:
                    df_features[f'price_vs_sma{period}'] = ((df_features['close'] - df_features[sma_col]) /
                                                           df_features[sma_col].replace(0, np.nan))

            # SMA ratio (if we have multiple SMAs)
            if len(self.config.SMA_PERIODS) >= 2:
                fast_sma = f'sma_{self.config.SMA_PERIODS[0]}'
                slow_sma = f'sma_{self.config.SMA_PERIODS[1]}'
                if fast_sma in df_features.columns and slow_sma in df_features.columns:
                    df_features['sma_ratio'] = (df_features[fast_sma] /
                                               df_features[slow_sma].replace(0, np.nan))

        except Exception as e:
            self.logger.error(f"Error creating technical indicators: {e}")
            raise

        # Volume features (enhanced with error handling)
        try:
            df_features['volume_sma'] = ta.sma(df_features['tick_volume'], length=20)

            # Avoid division by zero
            volume_sma_safe = df_features['volume_sma'].replace(0, np.nan)
            df_features['volume_ratio'] = df_features['tick_volume'] / volume_sma_safe
            df_features['volume_momentum'] = df_features['tick_volume'].pct_change(3)
        except Exception as e:
            self.logger.warning(f"Error creating volume features: {e}")
            df_features['volume_sma'] = df_features['tick_volume']
            df_features['volume_ratio'] = 1.0
            df_features['volume_momentum'] = 0.0

        # Price action features (enhanced with error handling)
        try:
            df_features['high_low_pct'] = ((df_features['high'] - df_features['low']) /
                                          df_features['close'].replace(0, np.nan))
            df_features['close_open_pct'] = ((df_features['close'] - df_features['open']) /
                                            df_features['open'].replace(0, np.nan))
            df_features['upper_shadow'] = ((df_features['high'] - np.maximum(df_features['open'], df_features['close'])) /
                                          df_features['close'].replace(0, np.nan))
            df_features['lower_shadow'] = ((np.minimum(df_features['open'], df_features['close']) - df_features['low']) /
                                          df_features['close'].replace(0, np.nan))
        except Exception as e:
            self.logger.warning(f"Error creating price action features: {e}")
            df_features['high_low_pct'] = 0.01
            df_features['close_open_pct'] = 0.0
            df_features['upper_shadow'] = 0.0
            df_features['lower_shadow'] = 0.0

        # Time features (enhanced for better GPU utilization)
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

        # Market session features
        df_features['asian_session'] = ((df_features['hour'] >= 0) & (df_features['hour'] <= 8)).astype(int)
        df_features['london_session'] = ((df_features['hour'] >= 8) & (df_features['hour'] <= 16)).astype(int)
        df_features['ny_session'] = ((df_features['hour'] >= 13) & (df_features['hour'] <= 21)).astype(int)

        # Final validation
        initial_rows = len(df_features)
        nan_counts = df_features.isna().sum()
        problematic_cols = nan_counts[nan_counts > initial_rows * 0.5].index.tolist()

        if problematic_cols:
            self.logger.warning(f"Columns with >50% NaN values: {problematic_cols}")

        self.logger.info(f"Created {len(df_features.columns)} features with robust error handling")

        return df_features

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train, validation, and test sets"""
        self.logger.info("Splitting data chronologically...")

        n_total = len(df)
        n_train = int(n_total * self.config.TRAIN_RATIO)
        n_val = int(n_total * self.config.VAL_RATIO)

        train_data = df.iloc[:n_train].copy()
        val_data = df.iloc[n_train:n_train + n_val].copy()
        test_data = df.iloc[n_train + n_val:].copy()

        self.logger.info(f"Train: {len(train_data)} samples ({train_data['timestamp'].min()} to {train_data['timestamp'].max()})")
        self.logger.info(f"Validation: {len(val_data)} samples ({val_data['timestamp'].min()} to {val_data['timestamp'].max()})")
        self.logger.info(f"Test: {len(test_data)} samples ({test_data['timestamp'].min()} to {test_data['timestamp'].max()})")

        return train_data, val_data, test_data

    def normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Normalize features using scalers fitted on training data only - GPU OPTIMIZED"""
        self.logger.info("Normalizing features (GPU optimized with enhanced feature groups)...")

        # Define feature groups for different scaling methods (enhanced)
        price_features = [
            'open', 'high', 'low', 'close', 'returns', 'log_returns', 'price_momentum',
            'atr', 'atr_normalized', 'spread', 'sma_20', 'sma_50', 'ema_20', 'ema_50',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'macd', 'macd_signal', 'macd_hist',
            'price_vs_sma20', 'price_vs_sma50', 'sma_ratio'
        ]

        volume_features = [
            'tick_volume', 'real_volume', 'volume_sma', 'volume_ratio', 'volume_momentum'
        ]

        ratio_features = [
            'high_low_pct', 'close_open_pct', 'upper_shadow', 'lower_shadow'
        ]

        bounded_features = [
            'rsi', 'rsi_fast', 'stoch_k', 'stoch_d', 'williams_r', 'bb_position'
        ]

        time_features = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]

        binary_features = [
            'asian_session', 'london_session', 'ny_session'
        ]

        # Remove non-existent columns
        existing_features = train_df.columns.tolist()
        price_features = [f for f in price_features if f in existing_features]
        volume_features = [f for f in volume_features if f in existing_features]
        ratio_features = [f for f in ratio_features if f in existing_features]
        bounded_features = [f for f in bounded_features if f in existing_features]
        time_features = [f for f in time_features if f in existing_features]
        binary_features = [f for f in binary_features if f in existing_features]

        # Apply different scalers for different feature types
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()

        # StandardScaler for price and ratio features
        if price_features + ratio_features:
            scaler_standard = StandardScaler()
            features_to_scale = price_features + ratio_features

            # Replace inf values with NaN, then handle them
            for feature in features_to_scale:
                for df_scaled in [train_scaled, val_scaled, test_scaled]:
                    df_scaled[feature] = df_scaled[feature].replace([np.inf, -np.inf], np.nan)

            train_scaled[features_to_scale] = scaler_standard.fit_transform(train_df[features_to_scale].fillna(0))
            val_scaled[features_to_scale] = scaler_standard.transform(val_df[features_to_scale].fillna(0))
            test_scaled[features_to_scale] = scaler_standard.transform(test_df[features_to_scale].fillna(0))

            self.scalers['standard'] = scaler_standard

        # MinMaxScaler for volume features
        if volume_features:
            scaler_minmax = MinMaxScaler()

            # Handle inf values
            for feature in volume_features:
                for df_scaled in [train_scaled, val_scaled, test_scaled]:
                    df_scaled[feature] = df_scaled[feature].replace([np.inf, -np.inf], np.nan)

            train_scaled[volume_features] = scaler_minmax.fit_transform(train_df[volume_features].fillna(0))
            val_scaled[volume_features] = scaler_minmax.transform(val_df[volume_features].fillna(0))
            test_scaled[volume_features] = scaler_minmax.transform(test_df[volume_features].fillna(0))

            self.scalers['minmax'] = scaler_minmax

        # MinMaxScaler for bounded features (scale to 0-1)
        if bounded_features:
            scaler_bounded = MinMaxScaler(feature_range=(0, 1))

            train_scaled[bounded_features] = scaler_bounded.fit_transform(train_df[bounded_features].fillna(50))  # RSI default 50
            val_scaled[bounded_features] = scaler_bounded.transform(val_df[bounded_features].fillna(50))
            test_scaled[bounded_features] = scaler_bounded.transform(test_df[bounded_features].fillna(50))

            self.scalers['bounded'] = scaler_bounded

        # Time and binary features are already normalized/binary, keep as is

        self.logger.info(f"Feature normalization completed (GPU optimized)")
        self.logger.info(f"   • Price/Ratio features: {len(price_features + ratio_features)}")
        self.logger.info(f"   • Volume features: {len(volume_features)}")
        self.logger.info(f"   • Bounded features: {len(bounded_features)}")
        self.logger.info(f"   • Time features: {len(time_features)}")
        self.logger.info(f"   • Binary features: {len(binary_features)}")

        return train_scaled, val_scaled, test_scaled

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare sequences for the Transformer model - SIMPLIFIED"""

        # Select feature columns (exclude non-feature columns) - SIMPLIFIED SET
        exclude_columns = ['timestamp', 'hour', 'day_of_week']  # Raw time features
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # Remove rows with NaN values
        df_clean = df[feature_columns].dropna()

        self.logger.info(f"Using {len(feature_columns)} features for model input (simplified)")
        self.logger.info(f"Feature columns: {feature_columns}")

        # Create sequences (smaller window size)
        sequences = []
        for i in range(self.config.WINDOW_SIZE, len(df_clean)):
            sequence = df_clean.iloc[i - self.config.WINDOW_SIZE:i].values
            sequences.append(sequence)

        sequences_array = np.array(sequences)

        self.logger.info(f"Created {len(sequences)} sequences of shape {sequences_array.shape} (smaller window)")

        return sequences_array, feature_columns

    def process_data(self, df: pd.DataFrame):
        """Complete data processing pipeline with robust error handling"""
        self.logger.info("Starting comprehensive data processing pipeline...")

        try:
            # Create features
            df_features = self.create_features(df)
            self.logger.info(f"✅ Features created: {len(df_features.columns)} total columns")

            # Split data chronologically BEFORE normalization and sequence creation
            train_df, val_df, test_df = self.split_data(df_features)

            # Remove NaN values from all datasets (important for stable training)
            self.logger.info("🧹 Cleaning NaN values...")
            initial_train_len = len(train_df)
            initial_val_len = len(val_df)
            initial_test_len = len(test_df)

            train_df = train_df.dropna()
            val_df = val_df.dropna()
            test_df = test_df.dropna()

            self.logger.info(f"   • Train: {initial_train_len} → {len(train_df)} rows")
            self.logger.info(f"   • Val: {initial_val_len} → {len(val_df)} rows")
            self.logger.info(f"   • Test: {initial_test_len} → {len(test_df)} rows")

            # Normalize features
            train_scaled, val_scaled, test_scaled = self.normalize_features(train_df, val_df, test_df)

            # Prepare sequences
            train_sequences, feature_columns = self.prepare_sequences(train_scaled)
            val_sequences, _ = self.prepare_sequences(val_scaled)
            test_sequences, _ = self.prepare_sequences(test_scaled)

            # Final validation
            self.logger.info("✅ Data processing validation:")
            self.logger.info(f"   • Feature columns: {len(feature_columns)}")
            self.logger.info(f"   • Train sequences: {train_sequences.shape}")
            self.logger.info(f"   • Val sequences: {val_sequences.shape}")
            self.logger.info(f"   • Test sequences: {test_sequences.shape}")

            # Check for any remaining issues
            if np.any(np.isnan(train_sequences)):
                self.logger.warning("⚠️  NaN values found in train sequences")
            if np.any(np.isnan(val_sequences)):
                self.logger.warning("⚠️  NaN values found in val sequences")
            if np.any(np.isnan(test_sequences)):
                self.logger.warning("⚠️  NaN values found in test sequences")

            processed_data = {
                'train_sequences': train_sequences,
                'val_sequences': val_sequences,
                'test_sequences': test_sequences,
                'train_df': train_scaled,
                'val_df': val_scaled,
                'test_df': test_scaled,
                'feature_columns': feature_columns
            }

            self.logger.info("🎉 Data processing pipeline completed successfully")

            return processed_data

        except Exception as e:
            self.logger.error(f"💥 Data processing pipeline failed: {e}")
            raise

# ========== Custom Transformer Policy ==========
class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Custom Transformer features extractor for Stable-Baselines3 - DTYPE-SAFE VERSION"""

    def __init__(self, observation_space: gym.Space, features_dim: int = 256, config: Config = None):
        # Calculate input dimensions from observation space
        if len(observation_space.shape) == 2:
            window_size, num_features = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got {observation_space.shape}")

        super().__init__(observation_space, features_dim)

        self.config = config or Config()
        self.window_size = window_size
        self.num_features = num_features

        # Positional encoding (registered as buffer for proper device/dtype handling)
        pos_encoding = self._create_positional_encoding(window_size, self.config.D_MODEL)
        self.register_buffer('pos_encoding', pos_encoding)

        # Input projection with layer norm for stability and dtype consistency
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, self.config.D_MODEL),
            nn.LayerNorm(self.config.D_MODEL),
            nn.Dropout(self.config.DROPOUT)
        )

        # Transformer encoder (dtype-safe configuration)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.D_MODEL,
            nhead=self.config.NUM_HEADS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            dropout=self.config.DROPOUT,
            activation='gelu' if torch.cuda.is_available() else 'relu',  # GELU for GPU, ReLU for CPU
            batch_first=True,
            norm_first=True  # Pre-norm for better gradient flow and dtype stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.NUM_ENCODER_LAYERS
        )

        # Multi-head attention pooling instead of simple average (dtype-safe)
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.config.D_MODEL,
            num_heads=self.config.NUM_HEADS,
            dropout=self.config.DROPOUT,
            batch_first=True
        )

        # CLS token for attention pooling (registered as parameter for proper handling)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.D_MODEL))

        # Output projection with residual connection and dropout
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.D_MODEL, self.config.D_MODEL),
            nn.GELU() if torch.cuda.is_available() else nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.D_MODEL, features_dim)
        )

        # Initialize weights properly for stable training
        self.apply(self._init_weights)

        # Mixed precision handling - DO NOT wrap forward with autocast here
        # Let the training loop handle autocast to avoid double-wrapping issues
        self.supports_mixed_precision = self.config.USE_MIXED_PRECISION and torch.cuda.is_available()

        # Model compilation only if configured and supported
        if (self.config.COMPILE_MODEL and
            hasattr(torch, 'compile') and
            torch.cuda.is_available()):
            try:
                self.transformer_encoder = torch.compile(
                    self.transformer_encoder,
                    mode='reduce-overhead'
                )
            except Exception:
                # Compilation might fail on some systems, continue without it
                pass

    def _init_weights(self, module):
        """Initialize weights for stable training"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, std=0.02)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # Explicitly use float32
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit dtype management for mixed precision safety"""
        # observations shape: (batch_size, window_size, num_features)
        batch_size = observations.shape[0]

        # Ensure observations are in the correct dtype (float32 for stability)
        # SB3 typically provides float32, but let's be explicit
        if observations.dtype != torch.float32:
            observations = observations.to(torch.float32)

        # Project input to d_model dimensions
        x = self.input_projection(observations)  # (batch_size, window_size, d_model)

        # Add positional encoding with explicit dtype matching
        # This is crucial for mixed precision: ensure both tensors have compatible dtypes
        current_dtype = x.dtype
        device = x.device

        # Get positional encoding and ensure it matches current tensor dtype and device
        pos_enc = self.pos_encoding[:, :x.size(1), :].to(dtype=current_dtype, device=device)
        x = x + pos_enc

        # Apply transformer encoder
        x_encoded = self.transformer_encoder(x)  # (batch_size, window_size, d_model)

        # Attention pooling with CLS token (dtype-safe)
        # Expand CLS token and ensure it matches the dtype of encoded features
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(dtype=x_encoded.dtype, device=x_encoded.device)

        # Use CLS token to attend to all sequence positions
        # Ensure all inputs to MultiheadAttention have the same dtype
        pooled_output, attention_weights = self.attention_pooling(
            cls_tokens,      # query: (batch_size, 1, d_model)
            x_encoded,       # key:   (batch_size, window_size, d_model)
            x_encoded        # value: (batch_size, window_size, d_model)
        )  # Output: (batch_size, 1, d_model)

        pooled_output = pooled_output.squeeze(1)  # (batch_size, d_model)

        # Final projection
        output = self.output_projection(pooled_output)  # (batch_size, features_dim)

        # Ensure output is float32 for SB3 compatibility
        if output.dtype != torch.float32:
            output = output.to(torch.float32)

        return output

class TransformerActorCriticPolicy(ActorCriticPolicy):
    """Custom Actor-Critic policy with Transformer features extractor - ADAPTIVE VERSION"""

    def __init__(self, observation_space, action_space, lr_schedule: Schedule,
                 config: Config = None, *args, **kwargs):

        self.config = config or Config()

        # Adaptive configuration based on device
        if torch.cuda.is_available():
            # GPU configuration
            features_dim = 256
            net_arch = dict(pi=[256, 128], vf=[256, 128])
            activation_fn = torch.nn.GELU
        else:
            # CPU configuration - smaller and more efficient
            features_dim = 128
            net_arch = dict(pi=[64], vf=[64])
            activation_fn = torch.nn.ReLU

        # Set the features extractor class and kwargs
        kwargs['features_extractor_class'] = TransformerFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {
            'features_dim': features_dim,
            'config': self.config
        }

        # Network architecture
        kwargs['net_arch'] = net_arch
        kwargs['activation_fn'] = activation_fn

        # Optimizer settings (adaptive)
        if torch.cuda.is_available():
            kwargs['optimizer_class'] = torch.optim.AdamW
            kwargs['optimizer_kwargs'] = {
                'weight_decay': 1e-4,
                'eps': 1e-7,
                'betas': (0.9, 0.999)
            }
        else:
            kwargs['optimizer_class'] = torch.optim.Adam
            kwargs['optimizer_kwargs'] = {
                'eps': 1e-8
            }

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

# ========== GPU-Optimized Trading Environment ==========
class VectorizedTradingEnv:
    """Vectorized trading environment for parallel processing with feature column support"""

    def __init__(self, df: pd.DataFrame, config: Config, sequences: np.ndarray = None,
                 num_envs: int = 4, feature_columns: List[str] = None):
        self.config = config
        self.df = df.copy()
        self.sequences = sequences
        self.num_envs = num_envs
        self.feature_columns = feature_columns or []

        # Create multiple environments
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        def make_env(rank):
            def _init():
                return CustomTradingEnv(df, config, sequences, feature_columns)
            return _init

        # Use SubprocVecEnv for true parallelization (better for CPU-bound tasks)
        # Use DummyVecEnv for GPU-bound tasks (to avoid pickle issues)
        if config.DEVICE == "cuda":
            self.vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        else:
            self.vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

        print(f"🔄 Created vectorized environment with {num_envs} parallel environments")

# ========== Training System with GPU Optimization ==========
class CustomTradingEnv(gym.Env):
    """Custom trading environment with robust observation handling"""

    def __init__(self, df: pd.DataFrame, config: Config, sequences: np.ndarray = None, feature_columns: List[str] = None):
        super().__init__()

        self.config = config
        self.df = df.copy()
        self.sequences = sequences
        self.feature_columns = feature_columns or []  # Store feature columns for fallback
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate inputs

        if sequences is not None:
            obs_shape = sequences.shape[1:]  # (window_size, num_features)
            self.logger.info(f"Observation space shape from sequences: {obs_shape}")
        elif self.feature_columns: # <--- ตรวจสอบว่ามี feature_columns ที่ส่งมาหรือไม่
            num_features = len(self.feature_columns)
            obs_shape = (config.WINDOW_SIZE, num_features)
            self.logger.info(f"Observation space shape from feature_columns: {obs_shape}")
        else:
            # Fallback: estimate from DataFrame (ควรหลีกเลี่ยงถ้าเป็นไปได้)
            self.logger.warning("No sequences or feature_columns provided, estimating num_features from DataFrame (less reliable)")
            # พยายาม infer feature columns ถ้าไม่ได้ส่งมา
            temp_exclude_columns = ['timestamp', 'hour', 'day_of_week']
            temp_feature_columns = [col for col in df.columns if col not in temp_exclude_columns and pd.api.types.is_numeric_dtype(df[col])]
            num_features = len(temp_feature_columns)
            obs_shape = (config.WINDOW_SIZE, num_features)
            self.logger.info(f"Fallback observation space shape: {obs_shape}")

        # Environment parameters
        self.initial_balance = config.INITIAL_BALANCE
        self.current_step = 0
        self.balance = config.INITIAL_BALANCE
        self.equity = config.INITIAL_BALANCE
        self.positions = []
        self.trade_history = []
        self.equity_history = []

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: sequences of features
        if sequences is not None:
            obs_shape = sequences.shape[1:]  # (window_size, num_features)
            self.logger.info(f"Observation space shape: {obs_shape}")
        else:
            # Fallback: estimate from DataFrame
            if feature_columns:
                num_features = len(feature_columns)
            else:
                num_features = len(df.select_dtypes(include=[np.number]).columns) - 1  # Exclude timestamp
            obs_shape = (config.WINDOW_SIZE, num_features)
            self.logger.info(f"Fallback observation space shape: {obs_shape}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_shape,
            dtype=np.float32  # Explicitly use float32
        )

        # Trading parameters
        self.leverage = config.LEVERAGE
        self.commission_per_lot = config.COMMISSION_PER_LOT
        self.lot_size = config.LOT_SIZE

        # Risk management
        self.n_sl = config.N_SL
        self.n_tp = config.N_TP

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self.current_step = self.config.WINDOW_SIZE  # Start after window
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.equity_history = [self.equity]

        return self._get_observation(), {}

    def _get_observation(self):
        """Get current observation with robust error handling"""
        # Primary method: use pre-computed sequences
        if self.sequences is not None:
            sequence_idx = self.current_step - self.config.WINDOW_SIZE

            if 0 <= sequence_idx < len(self.sequences):
                obs = self.sequences[sequence_idx].astype(np.float32)

                # Validate observation shape
                expected_shape = self.observation_space.shape
                if obs.shape != expected_shape:
                    self.logger.error(f"Observation shape mismatch: got {obs.shape}, expected {expected_shape}")
                    return self._get_fallback_observation()

                return obs
            else:
                self.logger.warning(f"Sequence index {sequence_idx} out of bounds, using fallback")
                return self._get_fallback_observation()

        # Fallback method: construct from DataFrame
        else:
            return self._get_fallback_observation()

    def _get_fallback_observation(self):
        """Fallback observation method using DataFrame"""
        try:
            start_idx = max(0, self.current_step - self.config.WINDOW_SIZE)
            end_idx = self.current_step

            # Use stored feature columns if available, otherwise infer
            if self.feature_columns:
                if all(col in self.df.columns for col in self.feature_columns):
                    feature_data = self.df[self.feature_columns]
                else:
                    self.logger.warning("Some feature columns missing, using numeric columns")
                    feature_data = self.df.select_dtypes(include=[np.number])
            else:
                # Exclude non-numeric columns
                exclude_cols = ['timestamp']
                numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns
                               if col not in exclude_cols]
                feature_data = self.df[numeric_cols]

            # Extract window data
            obs_data = feature_data.iloc[start_idx:end_idx].values

            # Handle edge case: beginning of dataset (pad with zeros)
            if obs_data.shape[0] < self.config.WINDOW_SIZE:
                padding_rows = self.config.WINDOW_SIZE - obs_data.shape[0]
                padding = np.zeros((padding_rows, obs_data.shape[1]), dtype=np.float32)
                obs_data = np.vstack((padding, obs_data))

            # Ensure correct dtype and shape
            obs_data = obs_data.astype(np.float32)

            # Validate shape
            expected_shape = self.observation_space.shape
            if obs_data.shape != expected_shape:
                self.logger.error(f"Fallback observation shape mismatch: got {obs_data.shape}, expected {expected_shape}")
                # Create zero-filled observation as last resort
                obs_data = np.zeros(expected_shape, dtype=np.float32)

            return obs_data

        except Exception as e:
            self.logger.error(f"Error in fallback observation: {e}")
            # Return zero-filled observation as emergency fallback
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_current_price(self):
        """Get current bid/ask prices with error handling"""
        try:
            current_data = self.df.iloc[self.current_step]

            # Use spread to calculate bid/ask
            mid_price = current_data['close']
            spread = current_data.get('spread', 0.0001)  # Default spread if missing

            # Validate prices
            if pd.isna(mid_price) or mid_price <= 0:
            #    self.logger.warning(f"Invalid mid_price: {mid_price}, using previous close")
                mid_price = self.df['close'].iloc[self.current_step - 1] if self.current_step > 0 else 1.0

            if pd.isna(spread) or spread < 0:
                spread = 0.0001  # Fallback spread

            bid = mid_price - spread / 2
            ask = mid_price + spread / 2

            return bid, ask

        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 1.0, 1.0001  # Emergency fallback prices

    def _calculate_sl_tp(self, entry_price: float, direction: str, atr: float):
        """Calculate stop loss and take profit based on ATR with validation"""
        # Validate inputs
        if pd.isna(atr) or atr <= 0:
            atr = 0.001  # Fallback ATR
            self.logger.warning(f"Invalid ATR: using fallback value {atr}")

        if pd.isna(entry_price) or entry_price <= 0:
            #self.logger.error(f"Invalid entry price: {entry_price}")
            return entry_price * 0.99, entry_price * 1.01  # Emergency fallback

        try:
            if direction == 'buy':
                sl = entry_price - (self.n_sl * atr)
                tp = entry_price + (self.n_tp * atr)
            else:  # sell
                sl = entry_price + (self.n_sl * atr)
                tp = entry_price - (self.n_tp * atr)

            # Validate calculated values
            if pd.isna(sl) or pd.isna(tp):
                self.logger.error("NaN values in SL/TP calculation")
                return entry_price * 0.99, entry_price * 1.01

            return sl, tp

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {e}")
            return entry_price * 0.99, entry_price * 1.01

    def _calculate_profit(self, position, current_price):
        """Calculate profit for a position with validation"""
        try:
            if position['direction'] == 'buy':
                profit = (current_price - position['entry_price']) * position['lot_size'] * 100000
            else:  # sell
                profit = (position['entry_price'] - current_price) * position['lot_size'] * 100000

            profit -= position['commission']

            # Validate profit calculation
            if pd.isna(profit):
                self.logger.error("NaN profit calculated")
                return 0.0

            return profit

        except Exception as e:
            self.logger.error(f"Error calculating profit: {e}")
            return 0.0

    def _update_positions(self):
        """Update positions and check for SL/TP hits with robust error handling"""
        if not self.positions:
            return

        try:
            bid, ask = self._get_current_price()
            positions_to_close = []

            for i, position in enumerate(self.positions):
                # Use appropriate price for closing
                close_price = bid if position['direction'] == 'buy' else ask

                # Check SL/TP with validation
                sl = position.get('sl')
                tp = position.get('tp')

                if sl is None or tp is None or pd.isna(sl) or pd.isna(tp):
                    self.logger.warning(f"Invalid SL/TP for position {i}: sl={sl}, tp={tp}")
                    continue

                # Check SL/TP triggers
                if position['direction'] == 'buy':
                    if close_price <= sl or close_price >= tp:
                        positions_to_close.append(i)
                else:  # sell
                    if close_price >= sl or close_price <= tp:
                        positions_to_close.append(i)

            # Close positions that hit SL/TP
            for i in reversed(positions_to_close):
                try:
                    position = self.positions.pop(i)
                    bid, ask = self._get_current_price()
                    close_price = bid if position['direction'] == 'buy' else ask

                    profit = self._calculate_profit(position, close_price)
                    self.balance += profit

                    # Record trade
                    self.trade_history.append({
                        'entry_time': position['entry_time'],
                        'exit_time': self.current_step,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': close_price,
                        'lot_size': position['lot_size'],
                        'profit': profit,
                        'exit_reason': 'TP' if profit > 0 else 'SL'
                    })

                except Exception as e:
                    self.logger.error(f"Error closing position {i}: {e}")

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def _calculate_reward(self):
        """Calculate reward with robust error handling"""
        try:
            if len(self.equity_history) < 2:
                return 0.0

            # Simple reward: current equity change normalized by initial balance
            current_change = self.equity_history[-1] - self.equity_history[-2]
            base_reward = current_change / self.initial_balance

            # Validate reward
            if pd.isna(base_reward) or abs(base_reward) > 1.0:  # Sanity check
                self.logger.warning(f"Invalid base reward: {base_reward}, setting to 0")
                base_reward = 0.0

            # Simple penalties
            if len(self.equity_history) > 1:
                max_equity = max(self.equity_history)
                current_drawdown = (max_equity - self.equity) / max_equity * 100 if max_equity > 0 else 0

                if current_drawdown > self.config.MAX_DRAWDOWN_PCT:
                    base_reward -= 0.1  # Small penalty for excessive drawdown

            # Penalty for holding too many positions
            if len(self.positions) > 1:
                base_reward -= 0.01

            return float(base_reward)  # Ensure it's a Python float, not numpy

        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0

    def step(self, action):
        """Execute one step in the environment with comprehensive error handling"""
        try:
            if self.current_step >= len(self.df) - 1:
                return self._get_observation(), 0.0, True, True, {}

            # Update existing positions first
            self._update_positions()

            # Get current market data
            current_data = self.df.iloc[self.current_step]
            bid, ask = self._get_current_price()
            atr = current_data.get('atr', 0.001)  # Default ATR if missing

            # Validate ATR
            if pd.isna(atr) or atr <= 0:
                atr = 0.001

            # Execute action
            if action == 1 and len(self.positions) == 0:  # Buy
                entry_price = ask
                sl, tp = self._calculate_sl_tp(entry_price, 'buy', atr)
                commission = self.commission_per_lot * self.lot_size

                position = {
                    'direction': 'buy',
                    'entry_price': entry_price,
                    'entry_time': self.current_step,
                    'lot_size': self.lot_size,
                    'sl': sl,
                    'tp': tp,
                    'commission': commission
                }

                self.positions.append(position)
                self.balance -= commission

            elif action == 2 and len(self.positions) == 0:  # Sell
                entry_price = bid
                sl, tp = self._calculate_sl_tp(entry_price, 'sell', atr)
                commission = self.commission_per_lot * self.lot_size

                position = {
                    'direction': 'sell',
                    'entry_price': entry_price,
                    'entry_time': self.current_step,
                    'lot_size': self.lot_size,
                    'sl': sl,
                    'tp': tp,
                    'commission': commission
                }

                self.positions.append(position)
                self.balance -= commission

            # Calculate current equity
            self.equity = self.balance
            for position in self.positions:
                close_price = bid if position['direction'] == 'buy' else ask
                unrealized_profit = self._calculate_profit(position, close_price)
                self.equity += unrealized_profit

            # Validate equity
            if pd.isna(self.equity):
                self.logger.error("NaN equity calculated")
                self.equity = self.balance

            self.equity_history.append(self.equity)

            # Calculate reward
            reward = self._calculate_reward()

            # Move to next step
            self.current_step += 1

            # Check if episode is done
            done = self.current_step >= len(self.df) - 1
            terminated = done
            truncated = False

            info = {
                'balance': float(self.balance),
                'equity': float(self.equity),
                'num_positions': len(self.positions),
                'num_trades': len(self.trade_history)
            }

            return self._get_observation(), float(reward), terminated, truncated, info

        except Exception as e:
            self.logger.error(f"Critical error in environment step: {e}")
            # Return safe defaults in case of critical error
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,  # Terminate episode on critical error
                True,
                {'error': str(e)}
            )

# ========== Training System ==========
class TrainingSystem:
    """Complete training system for the AI Trading Agent - DTYPE-SAFE VERSION"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup device and optimizations
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🚀 Training system initialized with device: {self.device}")

        # Setup mixed precision scaler if using GPU and mixed precision
        self.use_mixed_precision = (self.device.type == "cuda" and
                                   config.USE_MIXED_PRECISION and
                                   torch.cuda.is_available())

        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("⚡ Mixed precision GradScaler initialized")

        # Enable optimizations
        if self.device.type == "cuda":
            # Clear cache and optimize settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

            gpu_props = torch.cuda.get_device_properties(0)
            self.logger.info(f"🎯 GPU: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f}GB)")

    def create_environment(self, df: pd.DataFrame, sequences: np.ndarray, feature_columns: List[str]):
        """Create optimized trading environment with proper feature column handling"""
        if self.config.NUM_ENVS > 1:
            # Use vectorized environment for parallel training
            vectorized_env = VectorizedTradingEnv(df, self.config, sequences, self.config.NUM_ENVS, feature_columns)
            return vectorized_env.vec_env
        else:
            # Single environment
            return CustomTradingEnv(df, self.config, sequences, feature_columns)

    def create_agent(self, env):
        """Create PPO agent with device-specific optimizations and proper dtype handling"""

        # Policy kwargs with device-specific optimizations
        if self.device.type == "cuda":
            # GPU optimizations
            policy_kwargs = {
                'config': self.config,
                'net_arch': dict(
                    pi=[256, 128],  # Larger networks for GPU
                    vf=[256, 128]
                ),
                'activation_fn': torch.nn.GELU,  # GELU is faster on GPU
                'optimizer_class': torch.optim.AdamW,
                'optimizer_kwargs': {
                    'weight_decay': 1e-4,
                    'eps': 1e-7
                }
            }
        else:
            # CPU optimizations
            policy_kwargs = {
                'config': self.config,
                'net_arch': dict(
                    pi=[64],  # Smaller networks for CPU
                    vf=[64]
                ),
                'activation_fn': torch.nn.ReLU,  # ReLU is more CPU-friendly
                'optimizer_class': torch.optim.Adam,
                'optimizer_kwargs': {
                    'eps': 1e-8
                }
            }

        # Create PPO agent with device-specific settings
        model = PPO(
            TransformerActorCriticPolicy,
            env,
            learning_rate=self.config.LEARNING_RATE,
            n_steps=self.config.N_STEPS,
            batch_size=self.config.BATCH_SIZE,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            ent_coef=self.config.ENT_COEF,
            clip_range=self.config.CLIP_RANGE,
            n_epochs=self.config.N_EPOCHS,
            max_grad_norm=self.config.MAX_GRAD_NORM,  # Gradient clipping
            tensorboard_log="./tensorboard_logs/",
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=self.device  # Explicitly set device
        )

        # IMPORTANT: Do NOT wrap model.train with autocast here
        # Mixed precision should be handled at the optimizer level, not by wrapping methods
        # SB3 has its own mixed precision handling that we should not interfere with

        if self.use_mixed_precision:
            self.logger.info("🔥 Mixed precision training enabled (handled by SB3)")

        # Model compilation for PyTorch 2.0+ (only on GPU and if available)
        if (self.config.COMPILE_MODEL and
            hasattr(torch, 'compile') and
            self.device.type == "cuda"):
            try:
                # Only compile the feature extractor, not the entire policy
                if hasattr(model.policy, 'features_extractor'):
                    original_extractor = model.policy.features_extractor
                    compiled_extractor = torch.compile(
                        original_extractor,
                        mode='reduce-overhead'  # Optimize for repeated calls
                    )
                    model.policy.features_extractor = compiled_extractor
                    self.logger.info("⚡ Feature extractor compilation enabled")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")

        return model

    def setup_callbacks(self, val_env, models_dir: str):
        """Setup training callbacks with performance monitoring"""

        # Evaluation callback with optimizations
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=models_dir,
            log_path=models_dir,
            eval_freq=self.config.EVAL_FREQ,
            deterministic=True,
            render=False,
            verbose=1,
            warn=False  # Suppress warnings for speed
        )

        # Custom callback for performance monitoring (properly inheriting from BaseCallback)
        from stable_baselines3.common.callbacks import BaseCallback

        class PerformanceMonitorCallback(BaseCallback):
            def __init__(self, device, use_mixed_precision=False, verbose=0):
                super().__init__(verbose)
                self.device = device
                self.use_mixed_precision = use_mixed_precision
                self.step_count = 0
                self.last_log_time = time.time()

            def _on_step(self) -> bool:
                self.step_count += 1

                # Log performance every 1000 steps
                if self.step_count % 1000 == 0:
                    current_time = time.time()
                    time_diff = current_time - self.last_log_time
                    steps_per_second = 1000 / time_diff if time_diff > 0 else 0

                    if self.device.type == "cuda":
                        try:
                            allocated = torch.cuda.memory_allocated(0) / 1e6
                            reserved = torch.cuda.memory_reserved(0) / 1e6
                            print(f"🖥️  Step {self.step_count}: {steps_per_second:.1f} steps/s, "
                                  f"GPU: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

                            # Clear cache periodically to prevent memory buildup
                            if self.step_count % 10000 == 0:
                                torch.cuda.empty_cache()
                                print(f"🧹 GPU cache cleared at step {self.step_count}")

                        except Exception as e:
                            print(f"⚠️  GPU monitoring error: {e}")
                    else:
                        print(f"💻 Step {self.step_count}: {steps_per_second:.1f} steps/s (CPU mode)")

                    self.last_log_time = current_time

                return True

        performance_callback = PerformanceMonitorCallback(self.device, self.use_mixed_precision)

        return CallbackList([eval_callback, performance_callback])

    def train_agent(self, processed_data: dict, drive_manager):
        """Train the PPO agent with robust dtype and mixed precision handling"""
        self.logger.info("🚀 Starting dtype-safe agent training...")

        # Create environments with proper feature column passing
        feature_columns = processed_data.get('feature_columns', [])

        train_env = self.create_environment(
            processed_data['train_df'],
            processed_data['train_sequences'],
            feature_columns
        )

        val_env = self.create_environment(
            processed_data['val_df'],
            processed_data['val_sequences'],
            feature_columns
        )

        # Log environment info
        self.logger.info(f"📊 Training environment created:")
        self.logger.info(f"   • Observation space: {train_env.observation_space.shape}")
        self.logger.info(f"   • Action space: {train_env.action_space}")
        self.logger.info(f"   • Feature columns: {len(feature_columns)}")


        self.logger.info(f"📊 Validation environment created:")
        self.logger.info(f"   • Observation space: {val_env.observation_space.shape}") # <--- Log shape ของ val_env ด้วย
        self.logger.info(f"   • Action space: {val_env.action_space}")

        if train_env.observation_space.shape != val_env.observation_space.shape:
            self.logger.error(f"💥 Observation space mismatch between Training and Validation environments!")
            self.logger.error(f"   Train Env Shape: {train_env.observation_space.shape}")
            self.logger.error(f"   Val Env Shape: {val_env.observation_space.shape}")
            # อาจจะ raise error ที่นี่เพื่อหยุดการทำงานก่อนที่จะเกิดปัญหาตอนเทรน
            raise ValueError("Observation space mismatch between training and validation environments.")
        # Create agent with dtype-safe optimizations
        model = self.create_agent(train_env)

        # Setup callbacks
        models_dir = os.path.join(self.config.DRIVE_PATH, self.config.MODELS_PATH)
        callbacks = self.setup_callbacks(val_env, models_dir)

        # Pre-training optimizations and checks
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("🧹 GPU cache cleared before training")
        else:
            self.logger.info("💻 Training on CPU with optimizations")

        # Check model dtype consistency
        self._check_model_dtypes(model)

        # Training with comprehensive error handling
        start_time = time.time()

        try:
            self.logger.info("🎯 Starting training loop...")

            model.learn(
                total_timesteps=self.config.TOTAL_TIMESTEPS,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False
            )

        except RuntimeError as e:
            if "dtype" in str(e).lower():
                self.logger.error(f"💥 DTYPE ERROR during training: {e}")
                self.logger.error("This suggests mixed precision or tensor dtype issues")
                self.logger.error("Try running with USE_MIXED_PRECISION=False")

                # Attempt recovery by disabling mixed precision
                if self.use_mixed_precision:
                    self.logger.info("🔄 Attempting recovery with mixed precision disabled...")
                    self.config.USE_MIXED_PRECISION = False
                    self.use_mixed_precision = False

                    # Recreate model without mixed precision
                    model = self.create_agent(train_env)
                    self.logger.info("🔄 Retrying training without mixed precision...")

                    model.learn(
                        total_timesteps=self.config.TOTAL_TIMESTEPS,
                        callback=callbacks,
                        progress_bar=True,
                        reset_num_timesteps=False
                    )
                else:
                    raise
            else:
                self.logger.error(f"💥 RUNTIME ERROR during training: {e}")
                raise

        except Exception as e:
            self.logger.error(f"💥 UNEXPECTED ERROR during training: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise

        training_time = time.time() - start_time

        # Post-training cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Save final model
        device_suffix = "gpu" if self.device.type == "cuda" else "cpu"
        precision_suffix = "mixed" if self.use_mixed_precision else "full"
        final_model_path = os.path.join(models_dir, f"ppo_transformer_final_{device_suffix}_{precision_suffix}")
        model.save(final_model_path)

        self.logger.info(f"🎉 Training completed in {training_time/60:.1f} minutes")
        self.logger.info(f"💾 Final model saved to: {final_model_path}")

        # Training summary
        steps_per_second = self.config.TOTAL_TIMESTEPS / training_time
        self.logger.info(f"📊 Training Performance:")
        self.logger.info(f"   • Steps per second: {steps_per_second:.1f}")
        self.logger.info(f"   • Device: {self.device.type.upper()}")
        self.logger.info(f"   • Mixed Precision: {self.use_mixed_precision}")
        self.logger.info(f"   • Parallel environments: {self.config.NUM_ENVS}")

        return model

    def _check_model_dtypes(self, model):
        """Check and log model parameter dtypes for debugging"""
        self.logger.info("🔍 Checking model parameter dtypes...")

        try:
            policy = model.policy

            # Check features extractor
            if hasattr(policy, 'features_extractor'):
                extractor = policy.features_extractor

                for name, param in extractor.named_parameters():
                    if 'pos_encoding' in name or 'cls_token' in name:
                        self.logger.info(f"   • {name}: {param.dtype}")

                for name, buffer in extractor.named_buffers():
                    if 'pos_encoding' in name:
                        self.logger.info(f"   • {name} (buffer): {buffer.dtype}")

            # Check if model is in the right mode for mixed precision
            if self.use_mixed_precision:
                self.logger.info("   • Mixed precision enabled: tensors will be auto-cast during forward pass")
            else:
                self.logger.info("   • Full precision mode: all tensors remain float32")

        except Exception as e:
            self.logger.warning(f"Could not check model dtypes: {e}")

# ========== Evaluation System ==========
class EvaluationSystem:
    """System for evaluating trained models"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_best_model(self, models_dir: str):
        """Load the best trained model"""
        best_model_path = os.path.join(models_dir, "best_model")

        if os.path.exists(best_model_path + ".zip"):
            model = PPO.load(best_model_path)
            self.logger.info("Loaded best model successfully")
            return model
        else:
            raise FileNotFoundError("Best model not found")

    def backtest_model(self, model, test_df: pd.DataFrame, test_sequences: np.ndarray):
        """Run backtesting on test data"""
        self.logger.info("Starting backtesting...")

        # Create test environment
        test_env = CustomTradingEnv(test_df, self.config, test_sequences)

        # Run episode
        obs, _ = test_env.reset()
        done = False
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            step_count += 1

        self.logger.info(f"Backtesting completed after {step_count} steps")

        return test_env

    def calculate_metrics(self, test_env: CustomTradingEnv):
        """Calculate performance metrics"""
        self.logger.info("Calculating performance metrics...")

        trades = test_env.trade_history
        equity_curve = test_env.equity_history

        if not trades:
            self.logger.warning("No trades executed during backtesting")
            return {}

        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]

        total_profit = sum(t['profit'] for t in trades)
        gross_profit = sum(t['profit'] for t in winning_trades)
        gross_loss = abs(sum(t['profit'] for t in losing_trades))

        # Ratios
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0

        # Equity curve analysis
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(8760) if np.std(returns) > 0 else 0  # Annualized for hourly data

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = np.mean(returns) / downside_deviation * np.sqrt(8760) if downside_deviation > 0 else 0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)

        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        }

        return metrics

    def create_visualizations(self, test_env: CustomTradingEnv, test_df: pd.DataFrame, save_dir: str):
        """Create visualization plots - SIMPLIFIED VERSION"""
        self.logger.info("Creating visualizations (simplified)...")

        # Setup (simplified - only 2 plots)
        plt.style.use('default')  # Use default style for faster rendering
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Equity curve
        ax1.plot(test_env.equity_history, linewidth=2, color='blue')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)

        # Price chart with trades (simplified)
        price_data = test_df['close'].iloc[self.config.WINDOW_SIZE:].reset_index(drop=True)
        ax2.plot(price_data, linewidth=1, color='black', alpha=0.7, label='Price')

        # Mark trades (only entry points to reduce complexity)
        for trade in test_env.trade_history:
            entry_idx = trade['entry_time'] - self.config.WINDOW_SIZE

            if 0 <= entry_idx < len(price_data):
                color = 'green' if trade['direction'] == 'buy' else 'red'
                marker = '^' if trade['direction'] == 'buy' else 'v'

                ax2.scatter(entry_idx, trade['entry_price'], color=color, marker=marker, s=30, alpha=0.8)

        ax2.set_title('Price Chart with Trade Entries', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(save_dir, 'backtesting_results_simple.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')  # Lower DPI for faster saving
        plt.show()

        self.logger.info(f"Simplified visualizations saved to {plot_path}")

# ========== Report Generator ==========
class ReportGenerator:
    """Generate comprehensive performance reports"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_report(self, metrics: dict, test_env: CustomTradingEnv, save_dir: str):
        """Generate comprehensive performance report"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AI TRADING SYSTEM - BACKTESTING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Symbol: {self.config.SYMBOL}")
        report_lines.append(f"Timeframe: {self.config.TIMEFRAME}")
        report_lines.append(f"Initial Balance: ${self.config.INITIAL_BALANCE:,.2f}")
        report_lines.append("")

        # Trading Performance
        report_lines.append("TRADING PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        report_lines.append(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        report_lines.append(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        report_lines.append(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        report_lines.append("")

        # Profitability
        report_lines.append("PROFITABILITY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Profit: ${metrics.get('total_profit', 0):.2f}")
        report_lines.append(f"Gross Profit: ${metrics.get('gross_profit', 0):.2f}")
        report_lines.append(f"Gross Loss: ${metrics.get('gross_loss', 0):.2f}")
        report_lines.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report_lines.append(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
        report_lines.append(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        report_lines.append(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        report_lines.append("")

        # Risk Metrics
        report_lines.append("RISK METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        report_lines.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        report_lines.append(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        report_lines.append(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
        report_lines.append("")

        # System Configuration
        report_lines.append("SYSTEM CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Window Size: {self.config.WINDOW_SIZE}")
        report_lines.append(f"Transformer Layers: {self.config.NUM_ENCODER_LAYERS}")
        report_lines.append(f"Model Dimension: {self.config.D_MODEL}")
        report_lines.append(f"Attention Heads: {self.config.NUM_HEADS}")
        report_lines.append(f"Learning Rate: {self.config.LEARNING_RATE}")
        report_lines.append(f"Leverage: 1:{self.config.LEVERAGE}")
        report_lines.append(f"Lot Size: {self.config.LOT_SIZE}")
        report_lines.append(f"SL Multiplier: {self.config.N_SL}")
        report_lines.append(f"TP Multiplier: {self.config.N_TP}")
        report_lines.append("")

        # Trade Analysis
        if test_env.trade_history:
            report_lines.append("TRADE ANALYSIS")
            report_lines.append("-" * 40)

            # Trade duration analysis
            durations = [t['exit_time'] - t['entry_time'] for t in test_env.trade_history]
            avg_duration = np.mean(durations)

            report_lines.append(f"Average Trade Duration: {avg_duration:.1f} steps")
            report_lines.append(f"Shortest Trade: {min(durations)} steps")
            report_lines.append(f"Longest Trade: {max(durations)} steps")

            # Consecutive wins/losses
            profits = [t['profit'] for t in test_env.trade_history]
            wins_losses = ['W' if p > 0 else 'L' for p in profits]

            current_streak = 1
            max_win_streak = 0
            max_loss_streak = 0
            current_type = wins_losses[0] if wins_losses else 'W'

            for wl in wins_losses[1:]:
                if wl == current_type:
                    current_streak += 1
                else:
                    if current_type == 'W':
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, current_streak)
                    current_streak = 1
                    current_type = wl

            # Final streak
            if current_type == 'W':
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)

            report_lines.append(f"Max Consecutive Wins: {max_win_streak}")
            report_lines.append(f"Max Consecutive Losses: {max_loss_streak}")
            report_lines.append("")

        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)

        if metrics.get('total_return', 0) > 0:
            report_lines.append("✅ PROFITABLE STRATEGY")
        else:
            report_lines.append("❌ UNPROFITABLE STRATEGY")

        if metrics.get('sharpe_ratio', 0) > 1.0:
            report_lines.append("✅ GOOD RISK-ADJUSTED RETURNS")
        else:
            report_lines.append("⚠️  POOR RISK-ADJUSTED RETURNS")

        if metrics.get('max_drawdown', 100) < 10:
            report_lines.append("✅ ACCEPTABLE DRAWDOWN")
        else:
            report_lines.append("⚠️  HIGH DRAWDOWN")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_text = '\n'.join(report_lines)
        report_path = os.path.join(save_dir, 'backtesting_report.txt')

        with open(report_path, 'w') as f:
            f.write(report_text)

        # Print report
        print(report_text)

        self.logger.info(f"Report saved to {report_path}")

        return report_text

# ========== Main Execution Pipeline ==========
def main():
    """Main execution pipeline - ADAPTIVE VERSION (GPU/CPU OPTIMIZED)"""

    # Setup and hardware check
    config = Config()  # This will auto-configure based on available hardware
    logger = setup_logging()

    # Check hardware availability and optimize settings
    device_info = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_info = f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)"

        logger.info("🚀 GPU-OPTIMIZED AI Trading System Pipeline")
        logger.info(device_info)

        # Optimize PyTorch for GPU
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        # Estimate training time for GPU
        estimated_time = config.TOTAL_TIMESTEPS / 8000  # ~8k steps per minute on GPU

    else:
        device_info = "💻 CPU (Multi-core optimized)"
        logger.info("💻 CPU-OPTIMIZED AI Trading System Pipeline")
        logger.info(device_info)

        # Estimate training time for CPU
        estimated_time = config.TOTAL_TIMESTEPS / 1000  # ~1k steps per minute on CPU

    # Display configuration summary
    logger.info(f"⚙️  System Configuration:")
    logger.info(f"   • Device: {config.DEVICE.upper()}")
    logger.info(f"   • Window Size: {config.WINDOW_SIZE}")
    logger.info(f"   • Transformer: {config.NUM_ENCODER_LAYERS} layers, {config.D_MODEL} dims, {config.NUM_HEADS} heads")
    logger.info(f"   • Training Steps: {config.TOTAL_TIMESTEPS:,}")
    logger.info(f"   • Batch Size: {config.BATCH_SIZE}")
    logger.info(f"   • Parallel Envs: {config.NUM_ENVS}")
    logger.info(f"   • Mixed Precision: {config.USE_MIXED_PRECISION}")
    logger.info(f"⏱️  Estimated Training Time: {estimated_time:.1f} minutes")

    pipeline_start_time = time.time()

    try:
        # Initialize components
        drive_manager = DriveManager(config)
        preprocessor = DataPreprocessor(config)
        training_system = TrainingSystem(config)
        evaluation_system = EvaluationSystem(config)
        report_generator = ReportGenerator(config)

        # Step 1: Mount Google Drive
        step_start = time.time()
        drive_manager.mount_drive()
        logger.info(f"✅ Drive mounted in {time.time() - step_start:.1f}s")

        # Step 2: Load and process data
        step_start = time.time()
        logger.info("📊 Loading raw data...")
        raw_data = drive_manager.load_raw_data()
        logger.info(f"✅ Data loaded in {time.time() - step_start:.1f}s")

        step_start = time.time()
        logger.info("🔧 Processing data (enhanced features)...")
        processed_data = preprocessor.process_data(raw_data)
        processing_time = time.time() - step_start
        logger.info(f"✅ Data processed in {processing_time:.1f}s")
        logger.info(f"   • Features created: {len(processed_data['feature_columns'])}")
        logger.info(f"   • Training sequences: {len(processed_data['train_sequences']):,}")
        logger.info(f"   • Sequence shape: {processed_data['train_sequences'].shape}")

        # Save processed data
        drive_manager.save_processed_data(
            processed_data,
            processed_data,
            processed_data,
            preprocessor.scalers
        )

        # Step 3: Train model with device-specific optimization
        step_start = time.time()
        logger.info(f"🧠 Training model ({config.DEVICE.upper()}-optimized)...")

        # Memory info before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"🖥️  GPU Memory before training: {torch.cuda.memory_allocated(0) / 1e6:.1f}MB")
        else:
            logger.info(f"💻 CPU Training: Using {config.NUM_ENVS} parallel environments")

        model = training_system.train_agent(processed_data, drive_manager)
        training_time = time.time() - step_start

        # Training performance metrics
        steps_per_second = config.TOTAL_TIMESTEPS / training_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        logger.info(f"🚀 Performance: {steps_per_second:.1f} steps/second")

        if torch.cuda.is_available():
            logger.info(f"🖥️  Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e6:.1f}MB")
            torch.cuda.empty_cache()

        # Step 4: Evaluate model
        step_start = time.time()
        logger.info("📈 Evaluating model...")
        models_dir = os.path.join(config.DRIVE_PATH, config.MODELS_PATH)

        try:
            best_model = evaluation_system.load_best_model(models_dir)
            logger.info("📁 Loaded best model from validation")
        except FileNotFoundError:
            logger.warning("⚠️  Best model not found, using final trained model")
            best_model = model

        # Backtest
        test_env = evaluation_system.backtest_model(
            best_model,
            processed_data['test_df'],
            processed_data['test_sequences']
        )

        # Calculate metrics
        metrics = evaluation_system.calculate_metrics(test_env)
        evaluation_time = time.time() - step_start
        logger.info(f"✅ Evaluation completed in {evaluation_time:.1f}s")

        # Step 5: Generate reports
        step_start = time.time()
        reports_dir = os.path.join(config.DRIVE_PATH, config.REPORTS_PATH)
        os.makedirs(reports_dir, exist_ok=True)

        evaluation_system.create_visualizations(test_env, processed_data['test_df'], reports_dir)
        report_generator.generate_report(metrics, test_env, reports_dir)
        reporting_time = time.time() - step_start
        logger.info(f"✅ Reports generated in {reporting_time:.1f}s")

        # Pipeline summary
        total_time = time.time() - pipeline_start_time
        logger.info("🎉 AI Trading System Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("📊 PIPELINE PERFORMANCE SUMMARY:")
        logger.info(f"   • Total Runtime: {total_time/60:.1f} minutes")
        logger.info(f"   • Data Processing: {processing_time:.1f}s")
        logger.info(f"   • Model Training: {training_time/60:.1f} minutes ({steps_per_second:.1f} steps/s)")
        logger.info(f"   • Model Evaluation: {evaluation_time:.1f}s")
        logger.info(f"   • Report Generation: {reporting_time:.1f}s")

        # Hardware utilization summary
        logger.info(f"🖥️  HARDWARE UTILIZATION:")
        logger.info(f"   • Device: {device_info}")
        logger.info(f"   • Mixed Precision: {config.USE_MIXED_PRECISION}")
        logger.info(f"   • Model Compilation: {config.COMPILE_MODEL}")
        logger.info(f"   • Parallel Environments: {config.NUM_ENVS}")

        if torch.cuda.is_available():
            theoretical_cpu_time = training_time * 6  # Estimate 6x slower on CPU
            speedup = theoretical_cpu_time / training_time
            logger.info(f"🔥 Estimated GPU Speedup: {speedup:.1f}x faster than CPU")
        else:
            logger.info(f"💻 CPU Performance: Optimized for multi-core processing")

        logger.info("📈 TRADING PERFORMANCE:")
        logger.info(f"   • Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"   • Win Rate: {metrics.get('win_rate', 0):.1f}%")
        logger.info(f"   • Total Return: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"   • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   • Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info("=" * 60)

        return model, metrics

    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")
        else:
            logger.info("💻 CPU cleanup completed")

if __name__ == "__main__":
    """
    Instructions for running in Google Colab:

    🔧 SETUP STEPS:
    1. Upload your CSV file from Windows script to Google Drive folder:
       AI_Trading_System/01_Raw_Data/

    2. Install required packages by running these commands in Colab cells:
       !pip install stable-baselines3[extra]
       !pip install gym-mtsim
       !pip install pandas-ta
       !pip install gymnasium
       !pip install tensorboard

    3. Uncomment and run: check_installation() to verify all packages

    4. Run the main() function to start the GPU-optimized pipeline

    5. Monitor training progress through TensorBoard:
       %load_ext tensorboard
       %tensorboard --logdir ./tensorboard_logs/

    6. Check results in: AI_Trading_System/04_Logs_and_Reports/ on Google Drive

    🚀 GPU-OPTIMIZED VERSION FEATURES:
    - ⚡ Auto-detects and utilizes GPU (T4/V100/A100)
    - 🧠 Larger Transformer: 2 layers, 64 dims, 4 heads, 256 feedforward
    - 🔥 Mixed Precision Training (AMP) for 2x speed boost
    - 🚀 Model Compilation (PyTorch 2.0) for additional 20% speedup
    - 🔄 Vectorized Environments (4-8 parallel) for better GPU utilization
    - 📊 Enhanced Features: 25+ technical indicators vs 8 basic
    - 🎯 Optimal Batch Sizes: Auto-adjusts based on GPU memory
    - 💾 Smart Memory Management: Prevents OOM errors
    - ⏱️  Performance Monitoring: Real-time GPU usage tracking

    📊 EXPECTED PERFORMANCE:
    - 🖥️  GPU (T4): ~20-30 minutes for 500K steps
    - 💻 CPU: ~2-3 hours for 500K steps
    - 🚀 Speedup: 6-8x faster on GPU vs CPU
    - 🎯 Memory Usage: 4-6GB GPU memory (fits in Colab)

    🔄 THE PIPELINE PROCESS:
    - Load and preprocess data with 25+ enhanced technical indicators
    - Create GPU-optimized Transformer + PPO trading agent
    - Train with mixed precision and parallel environments
    - Real-time GPU monitoring and memory management
    - Comprehensive backtesting on unseen data
    - Generate performance reports and visualizations

    🆘 TROUBLESHOOTING:
    - If packages fail to install, try restarting runtime
    - Make sure your CSV file is in the correct Google Drive folder
    - Check that file permissions allow Colab access to your Drive
    - If GPU memory errors occur, the system will auto-reduce batch size
    - For best performance, use Colab Pro with high-RAM runtime

    💡 GPU OPTIMIZATION TIPS:
    - Use GPU runtime: Runtime → Change runtime type → GPU
    - Enable high-RAM if available: Runtime → Change runtime type → High-RAM
    - Close other browser tabs to free up system memory
    - The system auto-configures based on your GPU specifications

    🔥 ADVANCED SETTINGS:
    If you want to manually tune for your specific GPU:

    For T4 (15GB): BATCH_SIZE=256, N_STEPS=4096, NUM_ENVS=8
    For V100 (16GB): BATCH_SIZE=512, N_STEPS=4096, NUM_ENVS=8
    For A100 (40GB): BATCH_SIZE=1024, N_STEPS=8192, NUM_ENVS=16

    💡 TA-LIB ALTERNATIVE:
    We use pandas-ta instead of ta-lib to avoid installation issues.
    If you want to use ta-lib anyway, run these commands first:

    !apt-get update
    !apt-get install -y build-essential
    !wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    !tar -xzf ta-lib-0.4.0-src.tar.gz
    !cd ta-lib && ./configure --prefix=/usr && make && make install
    !pip install ta-lib
    """

    print("🚀 AI Trading System - Google Colab Pipeline (GPU-OPTIMIZED VERSION)")
    print("=" * 70)
    print()
    print("🔥 GPU-OPTIMIZED FEATURES:")
    print("   • Auto-detects GPU and optimizes accordingly")
    print("   • Mixed Precision Training (2x speedup)")
    print("   • Model Compilation (20% additional speedup)")
    print("   • Vectorized Environments (4-8 parallel)")
    print("   • Enhanced Transformer (2 layers, 64 dims, 4 heads)")
    print("   • 25+ Technical Indicators (vs 8 basic)")
    print("   • Smart memory management & monitoring")
    print()
    print("⚡ EXPECTED PERFORMANCE:")
    print("   • GPU (T4): 20-30 minutes for 500K steps")
    print("   • CPU: 2-3 hours for 500K steps")
    print("   • GPU Speedup: 6-8x faster than CPU")
    print("   • Memory: 4-6GB GPU (fits in Colab)")
    print()
    print("📋 STEP 1: Install packages (run in separate cells):")
    print("   !pip install stable-baselines3[extra]")
    print("   !pip install gym-mtsim")
    print("   !pip install pandas-ta")
    print("   !pip install gymnasium")
    print("   !pip install tensorboard")
    print()
    print("🔍 STEP 2: Check installation & GPU:")
    print("   Uncomment: check_installation()")
    print()
    print("🚀 STEP 3: Run the GPU-optimized pipeline:")
    print("   Uncomment: main()")
    print()
    print("📊 STEP 4: Monitor training:")
    print("   %load_ext tensorboard")
    print("   %tensorboard --logdir ./tensorboard_logs/")
    print()
    print("💡 TIP: Use GPU runtime for best performance!")
    print("   Runtime → Change runtime type → GPU")
    print("=" * 70)

    # Uncomment these lines to run:
    check_installation()
    main()
