#!/usr/bin/env python3
"""
High-Performance Teacher Model Training Script
Optimized for large-scale phoneme recognition with knowledge distillation preparation
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from tqdm import tqdm

class SimpleConfig:
    """Simple config class for command line usage"""
    def __init__(self):
        self.DATA_ROOT = Path("datasets_original/en")
        self.FEATURE_DIR = self.DATA_ROOT / "features_clean"
        self.LABEL_DIR = self.DATA_ROOT / "labels_clean"
        self.MODEL_OUTPUT = Path("models/teacher_phoneme_model")
        self.LOGS_DIR = Path("logs/teacher_training")
        self.RESULTS_DIR = Path("results/teacher_model")
        self.SEGMENT_LENGTH = 100  # Reduced for CPU efficiency
        self.INPUT_DIM = 39
        self.BATCH_SIZE = 16  # Reduced for CPU efficiency
        self.EPOCHS = 100
        self.LEARNING_RATE = 1e-3
        self.VALIDATION_SPLIT = 0.15
        self.TEST_SPLIT = 0.1
        self.USE_MIXED_PRECISION = False  # Disabled for CPU
        self.CONV_FILTERS = [64, 128, 128]  # Reduced for CPU efficiency
        self.CONV_KERNELS = [7, 5, 3]
        self.GRU_UNITS = [256, 256, 128]  # Reduced for CPU efficiency
        self.DENSE_UNITS = [128, 64]  # Reduced for CPU efficiency
        self.DROPOUT_RATE = 0.3
        self.EARLY_STOPPING_PATIENCE = 15
        self.REDUCE_LR_PATIENCE = 5
        self.REDUCE_LR_FACTOR = 0.5

class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to show tqdm progress bar for epochs"""
    
    def __init__(self, verbose=1):
        super().__init__()
        self.verbose = verbose
        self.epoch_bar = None
        
    def on_train_begin(self, logs=None):
        if self.verbose:
            print("\n[TRAINING] Starting epoch progression:")
            self.epoch_bar = tqdm(total=self.params['epochs'], desc="Epochs", unit="epoch", 
                                position=0, leave=True, ncols=100)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose and self.epoch_bar:
            # Create clean metrics display
            metrics_info = []
            if logs:
                # Training metrics
                if 'loss' in logs:
                    metrics_info.append(f"loss: {logs['loss']:.4f}")
                if 'sparse_categorical_accuracy' in logs:
                    metrics_info.append(f"acc: {logs['sparse_categorical_accuracy']:.4f}")
                
                # Validation metrics
                if 'val_loss' in logs:
                    metrics_info.append(f"val_loss: {logs['val_loss']:.4f}")
                if 'val_sparse_categorical_accuracy' in logs:
                    metrics_info.append(f"val_acc: {logs['val_sparse_categorical_accuracy']:.4f}")
                
                # Learning rate if available
                if 'lr' in logs:
                    metrics_info.append(f"lr: {logs['lr']:.2e}")
            
            # Update progress bar with metrics
            metrics_str = " | ".join(metrics_info)
            self.epoch_bar.set_postfix_str(metrics_str)
            self.epoch_bar.update(1)
            
            # Print epoch summary line
            epoch_summary = f"Epoch {epoch+1:3d}/{self.params['epochs']}: {metrics_str}"
            tqdm.write(epoch_summary)
    
    def on_train_end(self, logs=None):
        if self.verbose and self.epoch_bar:
            self.epoch_bar.close()
            print("\n[TRAINING] Epoch progression completed!")
from tqdm import tqdm

# Enable mixed precision for performance (only if GPU available)
from tensorflow.keras import mixed_precision

# Configure GPU first
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Found {len(gpus)} GPU(s), memory growth enabled")
        # Only enable mixed precision if GPU is available
        mixed_precision.set_global_policy('mixed_float16')
        print("[GPU] Mixed precision enabled (float16)")
    except RuntimeError as e:
        print(f"[WARNING] GPU configuration error: {e}")
        print("[CPU] Using CPU only - disabling mixed precision")
else:
    print("[CPU] No GPU found - using CPU only")
    print("[CPU] Mixed precision disabled for CPU compatibility")

class TeacherModelTrainer:
    """High-performance teacher model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        # Data statistics
        self.feature_mean = None
        self.feature_std = None
        self.class_weights = None
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOGS_DIR / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.MODEL_OUTPUT, self.config.LOGS_DIR, self.config.RESULTS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load and prepare training data with performance optimizations"""
        self.logger.info("[INFO] Loading and preparing data...")
        start_time = time.time()
        
        # Find all available files
        feature_files = list(self.config.FEATURE_DIR.glob("*.npy"))
        label_files = list(self.config.LABEL_DIR.glob("*.txt"))
        
        # Get matching files
        feature_bases = {f.stem for f in feature_files}
        label_bases = {f.stem for f in label_files}
        matching_bases = feature_bases & label_bases
        
        self.logger.info(f"[STATS] Found {len(matching_bases)} matching feature-label pairs")
        
        if not matching_bases:
            raise ValueError("No matching feature-label pairs found!")
        
        # Collect all labels first for encoder fitting
        all_labels = []
        valid_files = []
        
        for base in matching_bases:
            label_file = self.config.LABEL_DIR / f"{base}.txt"
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    labels = [line.strip() for line in f.readlines()]
                    
                # Filter out OOV if still present
                labels = [label for label in labels if label != 'OOV']
                if len(labels) > self.config.SEGMENT_LENGTH:  # Only use files with sufficient length
                    all_labels.extend(labels)
                    valid_files.append(base)
            except Exception as e:
                self.logger.warning(f"Error reading {label_file}: {e}")
        
        # Fit label encoder
        distinct_phonemes = sorted(set(all_labels))
        self.label_encoder.fit(distinct_phonemes)
        self.logger.info(f"[PHONEMES] Found {len(distinct_phonemes)} distinct phonemes: {distinct_phonemes}")
        
        # Load features and create segments
        X_segments, y_segments = [], []
        skipped_files = 0
        
        for base in valid_files:
            try:
                # Load features
                features = np.load(self.config.FEATURE_DIR / f"{base}.npy")
                
                # Load labels
                with open(self.config.LABEL_DIR / f"{base}.txt", 'r', encoding='utf-8') as f:
                    labels = [line.strip() for line in f.readlines() if line.strip() != 'OOV']
                
                # Verify alignment
                if len(features) != len(labels):
                    self.logger.warning(f"Length mismatch in {base}: features={len(features)}, labels={len(labels)}")
                    skipped_files += 1
                    continue
                
                # Encode labels
                labels_encoded = self.label_encoder.transform(labels)
                
                # Create segments
                X_seg, y_seg = self._create_segments(features, labels_encoded)
                X_segments.extend(X_seg)
                y_segments.extend(y_seg)
                
            except Exception as e:
                self.logger.warning(f"Error processing {base}: {e}")
                skipped_files += 1
                continue
        
        if not X_segments:
            raise ValueError("No valid segments created!")
        
        # Convert to arrays with proper shapes for frame-level prediction
        X_all = np.array(X_segments, dtype=np.float32)
        y_all = np.array(y_segments, dtype=np.int32)
        
        # Ensure y_all has correct shape for TimeDistributed layers
        # Shape should be (samples, sequence_length) for sparse_categorical_crossentropy
        self.logger.info(f"[SHAPES] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
        
        # Compute normalization statistics
        self._compute_normalization_stats(X_all)
        
        # Normalize features
        X_all = (X_all - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Compute class weights
        self._compute_class_weights(y_all)
        
        data_stats = {
            'total_segments': len(X_all),
            'segment_length': self.config.SEGMENT_LENGTH,
            'feature_dim': self.config.INPUT_DIM,
            'num_classes': len(distinct_phonemes),
            'skipped_files': skipped_files,
            'phonemes': distinct_phonemes
        }
        
        load_time = time.time() - start_time
        self.logger.info(f"[SUCCESS] Data preparation completed in {load_time:.2f}s")
        self.logger.info(f"[STATS] Created {len(X_all)} segments from {len(valid_files)} files")
        
        return X_all, y_all, data_stats
    
    def _create_segments(self, features: np.ndarray, labels: np.ndarray) -> Tuple[List, List]:
        """Create overlapping segments for better data utilization"""
        segments_X, segments_y = [], []
        
        # Use overlapping windows for more training data
        stride = self.config.SEGMENT_LENGTH // 2  # 50% overlap
        
        for start in range(0, len(features) - self.config.SEGMENT_LENGTH + 1, stride):
            end = start + self.config.SEGMENT_LENGTH
            segments_X.append(features[start:end])
            segments_y.append(labels[start:end])
        
        return segments_X, segments_y
    
    def _compute_normalization_stats(self, X: np.ndarray):
        """Compute global normalization statistics"""
        # Reshape to (total_frames, features)
        X_flat = X.reshape(-1, X.shape[-1])
        self.feature_mean = np.mean(X_flat, axis=0).astype(np.float32)
        self.feature_std = np.std(X_flat, axis=0).astype(np.float32)
        
        # Save statistics
        self.config.MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
        np.save(self.config.MODEL_OUTPUT / "feature_mean.npy", self.feature_mean)
        np.save(self.config.MODEL_OUTPUT / "feature_std.npy", self.feature_std)
        
        self.logger.info(f"[STATS] Normalization stats computed and saved")
    
    def _compute_class_weights(self, y: np.ndarray):
        """Compute balanced class weights (currently disabled for TimeDistributed compatibility)"""
        # Note: class_weight not used with TimeDistributed layers due to shape incompatibility
        self.class_weights = None
        self.logger.info(f"[WEIGHTS] Class weights disabled for TimeDistributed compatibility")
    
    def build_model(self, num_classes: int) -> tf.keras.Model:
        """Build optimized teacher model architecture"""
        self.logger.info("[MODEL] Building teacher model...")
        
        # Input layer
        inputs = tf.keras.Input(
            shape=(self.config.SEGMENT_LENGTH, self.config.INPUT_DIM),
            name="audio_features"
        )
        
        x = inputs
        
        # Convolutional layers for local feature extraction
        for i, (filters, kernel) in enumerate(zip(self.config.CONV_FILTERS, self.config.CONV_KERNELS)):
            x = tf.keras.layers.Conv1D(
                filters, kernel, padding='same',
                activation='relu', name=f'conv1d_{i+1}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = tf.keras.layers.Dropout(self.config.DROPOUT_RATE / 2, name=f'dropout_conv_{i+1}')(x)
        
        # Recurrent layers for temporal modeling
        for i, units in enumerate(self.config.GRU_UNITS):
            return_sequences = i < len(self.config.GRU_UNITS) - 1 or True  # Always return sequences for frame-level prediction
            x = tf.keras.layers.GRU(
                units, return_sequences=return_sequences,
                dropout=self.config.DROPOUT_RATE,
                recurrent_dropout=self.config.DROPOUT_RATE / 2,
                name=f'gru_{i+1}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'bn_gru_{i+1}')(x)
        
        # Dense layers for classification
        for i, units in enumerate(self.config.DENSE_UNITS):
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units, activation='relu'),
                name=f'dense_{i+1}'
            )(x)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dropout(self.config.DROPOUT_RATE),
                name=f'dropout_dense_{i+1}'
            )(x)
        
        # Output layer
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax'),
            name='phoneme_predictions'
        )(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='teacher_phoneme_model')
        
        # Compile with mixed precision optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE,
            epsilon=1e-7  # Important for mixed precision
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']  # Removed sparse_top_k_categorical_accuracy (incompatible with mixed precision)
        )
        
        # Save model architecture
        self.config.MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
        with open(self.config.MODEL_OUTPUT / "model_architecture.json", 'w') as f:
            json.dump(model.to_json(), f, indent=2)
        
        self.logger.info(f"[SUCCESS] Model built with {model.count_params():,} parameters")
        return model
    
    def create_data_pipeline(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create optimized data pipeline"""
        self.logger.info("[PIPELINE] Creating optimized data pipeline...")
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        # Split data
        n_total = len(X)
        n_test = int(n_total * self.config.TEST_SPLIT)
        n_val = int(n_total * self.config.VALIDATION_SPLIT)
        n_train = n_total - n_test - n_val
        
        X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
        y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
        
        self.logger.info(f"[SPLIT] Data split: Train={n_train}, Val={n_val}, Test={n_test}")
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Optimize training dataset for CPU
        train_ds = (train_ds
                   .shuffle(buffer_size=min(1000, n_train))  # Smaller buffer for CPU
                   .batch(self.config.BATCH_SIZE, drop_remainder=True)
                   .prefetch(2))  # Conservative prefetching for CPU
        
        # Optimize validation dataset
        val_ds = (val_ds
                 .batch(self.config.BATCH_SIZE)
                 .prefetch(1))
        
        # Optimize test dataset
        test_ds = (test_ds
                  .batch(self.config.BATCH_SIZE)
                  .prefetch(1))
        
        return train_ds, val_ds, test_ds
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.REDUCE_LR_FACTOR,
                patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.MODEL_OUTPUT / "checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.config.LOGS_DIR),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            tf.keras.callbacks.CSVLogger(
                filename=str(self.config.LOGS_DIR / "training_history.csv"),
                append=False
            ),
            TqdmProgressCallback(verbose=1)  # Add tqdm progress bar
        ]
        
        return callbacks
    
    def train(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        """Train the teacher model"""
        self.logger.info("[TRAINING] Starting teacher model training...")
        
        # Build model
        self.model = self.build_model(num_classes)
        
        # Create data pipeline
        train_ds, val_ds, test_ds = self.create_data_pipeline(X, y)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            verbose=0  # Disable Keras progress bar (we use tqdm instead)
        )
        
        training_time = time.time() - start_time
        self.logger.info(f"[SUCCESS] Training completed in {training_time/3600:.2f} hours")
        
        # Evaluate on test set
        self.logger.info("[EVAL] Evaluating on test set...")
        test_results = self.model.evaluate(test_ds, verbose=1)
        
        # Save final model
        self.model.save(str(self.config.MODEL_OUTPUT / "teacher_model_final.h5"))
        self.model.save(str(self.config.MODEL_OUTPUT / "teacher_model_final"), save_format="tf")
        
        # Save training artifacts
        self._save_training_artifacts(test_results, training_time)
        
        return self.history, test_results
    
    def _save_training_artifacts(self, test_results: List[float], training_time: float):
        """Save training artifacts and metadata"""
        # Save label encoder
        with open(self.config.MODEL_OUTPUT / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save training metadata
        metadata = {
            'model_name': 'teacher_phoneme_model',
            'training_time_hours': training_time / 3600,
            'num_classes': len(self.label_encoder.classes_),
            'phonemes': self.label_encoder.classes_.tolist(),
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'config': {
                'segment_length': self.config.SEGMENT_LENGTH,
                'input_dim': self.config.INPUT_DIM,
                'batch_size': self.config.BATCH_SIZE,
                'epochs': self.config.EPOCHS,
                'learning_rate': self.config.LEARNING_RATE
            }
        }
        
        with open(self.config.MODEL_OUTPUT / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info("[SAVE] Training artifacts saved")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Teacher Phoneme Recognition Model')
    parser.add_argument('--data-root', type=str, default='datasets_original/en',
                       help='Root directory containing features_clean and labels_clean')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--segment-length', type=int, default=128,
                       help='Sequence segment length')
    
    args = parser.parse_args()
    
    # Update config with arguments
    config = SimpleConfig()
    config.DATA_ROOT = Path(args.data_root)
    config.FEATURE_DIR = config.DATA_ROOT / "features_clean"
    config.LABEL_DIR = config.DATA_ROOT / "labels_clean"
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.SEGMENT_LENGTH = args.segment_length
    
    # Initialize trainer
    trainer = TeacherModelTrainer(config)
    
    try:
        # Load and prepare data
        X, y, data_stats = trainer.load_and_prepare_data()
        
        # Train model
        history, test_results = trainer.train(X, y, data_stats['num_classes'])
        
        print("\n" + "="*60)
        print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"[RESULTS] Test Loss: {test_results[0]:.4f}")
        print(f"[RESULTS] Test Accuracy: {test_results[1]:.4f}")
        print(f"[OUTPUT] Model saved to: {config.MODEL_OUTPUT}")
        print("="*60)
        
    except Exception as e:
        trainer.logger.error(f"[ERROR] Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
