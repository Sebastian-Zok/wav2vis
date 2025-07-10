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
        self.SEGMENT_LENGTH = 96  # Balanced for RTX 3090
        self.INPUT_DIM = 39
        self.BATCH_SIZE = 24  # Optimized for 24GB GPU
        self.EPOCHS = 100  # Sufficient for convergence
        self.LEARNING_RATE = 1e-3  # Standard learning rate
        self.VALIDATION_SPLIT = 0.15
        self.TEST_SPLIT = 0.1
        self.USE_MIXED_PRECISION = True  # Enable for GPU memory efficiency
        # Teacher model architecture - balanced for memory and accuracy
        self.CONV_FILTERS = [96, 192, 192]  # Increased for RTX 3090
        self.CONV_KERNELS = [5, 3, 3]  # Smaller kernels
        self.GRU_UNITS = [384, 192]  # Increased units
        self.DENSE_UNITS = [192, 96]  # Larger dense layers
        self.DROPOUT_RATE = 0.3  # Prevent overfitting
        self.EARLY_STOPPING_PATIENCE = 15
        self.REDUCE_LR_PATIENCE = 5
        self.REDUCE_LR_FACTOR = 0.5
        # Checkpoint settings
        self.CHECKPOINT_FREQUENCY = 5
        self.SAVE_BEST_ONLY = True
        self.PROGRESS_UPDATE_FREQ = 10
        # For knowledge distillation - teacher should output soft probabilities
        self.TEMPERATURE = 4.0  # For soft targets in distillation
        self.SAVE_SOFT_TARGETS = True  # Save teacher predictions for student training
        # Memory optimization
        self.MAX_SEGMENTS = 200000  # Increased for RTX 3090
        self.SAMPLE_RATIO = 0.5  # Use 50% of available files for training

class EnhancedProgressCallback(tf.keras.callbacks.Callback):
    """Enhanced callback for detailed training progress with batch-level updates"""
    
    def __init__(self, verbose=1, update_freq=10):
        super().__init__()
        self.verbose = verbose
        self.update_freq = update_freq
        self.epoch_bar = None
        self.batch_bar = None
        self.current_epoch = 0
        self.steps_per_epoch = 0
        
    def on_train_begin(self, logs=None):
        if self.verbose:
            self.steps_per_epoch = self.params.get('steps', 0)
            print(f"\n{'='*80}")
            print(f"[TRAINING STARTED] Total Epochs: {self.params['epochs']}")
            print(f"[TRAINING STARTED] Steps per Epoch: {self.steps_per_epoch}")
            print(f"{'='*80}")
            
    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            self.current_epoch = epoch
            print(f"\n[EPOCH {epoch+1}/{self.params['epochs']}] Starting...")
            if self.steps_per_epoch > 0:
                self.batch_bar = tqdm(
                    total=self.steps_per_epoch, 
                    desc=f"Epoch {epoch+1:3d}", 
                    unit="batch",
                    ncols=120,
                    leave=False
                )
    
    def on_batch_end(self, batch, logs=None):
        if self.verbose and self.batch_bar and batch % self.update_freq == 0:
            metrics_info = []
            if logs:
                if 'loss' in logs:
                    metrics_info.append(f"loss: {logs['loss']:.4f}")
                if 'sparse_categorical_accuracy' in logs:
                    metrics_info.append(f"acc: {logs['sparse_categorical_accuracy']:.3f}")
            
            if metrics_info:
                self.batch_bar.set_postfix_str(" | ".join(metrics_info))
            self.batch_bar.update(self.update_freq)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self.batch_bar:
                self.batch_bar.close()
            
            # Print comprehensive epoch summary
            print(f"\n[EPOCH {epoch+1:3d} COMPLETED]")
            if logs:
                train_metrics = []
                val_metrics = []
                
                # Training metrics
                if 'loss' in logs:
                    train_metrics.append(f"Loss: {logs['loss']:.4f}")
                if 'sparse_categorical_accuracy' in logs:
                    train_metrics.append(f"Accuracy: {logs['sparse_categorical_accuracy']:.4f}")
                
                # Validation metrics
                if 'val_loss' in logs:
                    val_metrics.append(f"Val Loss: {logs['val_loss']:.4f}")
                if 'val_sparse_categorical_accuracy' in logs:
                    val_metrics.append(f"Val Accuracy: {logs['val_sparse_categorical_accuracy']:.4f}")
                
                # Learning rate
                lr_info = ""
                if 'lr' in logs:
                    lr_info = f" | LR: {logs['lr']:.2e}"
                
                print(f"  Training  -> {' | '.join(train_metrics)}")
                if val_metrics:
                    print(f"  Validation-> {' | '.join(val_metrics)}")
                if lr_info:
                    print(f"  Learning Rate{lr_info}")
            
            print(f"{'─'*60}")
    
    def on_train_end(self, logs=None):
        if self.verbose:
            print(f"\n{'='*80}")
            print("[TRAINING COMPLETED] All epochs finished!")
            print(f"{'='*80}")

# Enable mixed precision for performance (only if GPU available)
from tensorflow.keras import mixed_precision

# Configure GPU first
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Found {len(gpus)} GPU(s), memory growth enabled")
        # Enable mixed precision for memory efficiency
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
        
        # Get matching files with sampling for memory efficiency
        feature_bases = {f.stem for f in feature_files}
        label_bases = {f.stem for f in label_files}
        matching_bases = feature_bases & label_bases
        
        # Sample files to reduce memory usage
        if len(matching_bases) > int(len(matching_bases) * self.config.SAMPLE_RATIO):
            matching_bases = set(np.random.choice(
                list(matching_bases), 
                int(len(matching_bases) * self.config.SAMPLE_RATIO), 
                replace=False
            ))
        
        self.logger.info(f"[STATS] Using {len(matching_bases)} of available files (sample ratio: {self.config.SAMPLE_RATIO})")
        self.logger.info(f"[STATS] Found {len(matching_bases)} matching feature-label pairs")
        
        print(f"PATHAPASHJAFIKASJ: {self.config.FEATURE_DIR}")

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
                
                # Check if we've reached the segment limit
                if len(X_segments) >= self.config.MAX_SEGMENTS:
                    self.logger.info(f"[LIMIT] Reached maximum segments ({self.config.MAX_SEGMENTS}), stopping data loading")
                    break
                
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
        """Create overlapping segments with multiple stride patterns for better data utilization"""
        segments_X, segments_y = [], []
        
        # Use smaller stride patterns for memory efficiency
        strides = [
            self.config.SEGMENT_LENGTH // 2,   # 50% overlap
            self.config.SEGMENT_LENGTH,        # No overlap
        ]
        
        for stride in strides:
            for start in range(0, len(features) - self.config.SEGMENT_LENGTH + 1, stride):
                end = start + self.config.SEGMENT_LENGTH
                
                # Only include segments with sufficient non-silence content
                segment_labels = labels[start:end]
                non_silence_ratio = np.sum(segment_labels != self.label_encoder.transform(['SIL'])[0]) / len(segment_labels)
                
                if non_silence_ratio > 0.3:  # At least 30% non-silence
                    segments_X.append(features[start:end])
                    segments_y.append(segment_labels)
        
        # Remove random segments for memory efficiency
        # Just use the systematic segments above
        
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
        """Build teacher model optimized for knowledge distillation"""
        self.logger.info("[MODEL] Building teacher model for knowledge distillation...")
        
        # Input layer
        inputs = tf.keras.Input(
            shape=(self.config.SEGMENT_LENGTH, self.config.INPUT_DIM),
            name="audio_features"
        )
        
        x = inputs
        
        # Convolutional layers for feature extraction
        for i, (filters, kernel) in enumerate(zip(self.config.CONV_FILTERS, self.config.CONV_KERNELS)):
            x = tf.keras.layers.Conv1D(
                filters, kernel, padding='same',
                activation='relu', name=f'conv1d_{i+1}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = tf.keras.layers.Dropout(self.config.DROPOUT_RATE, name=f'dropout_conv_{i+1}')(x)
        
        # Bidirectional GRU layers for temporal modeling
        for i, units in enumerate(self.config.GRU_UNITS):
            if i == len(self.config.GRU_UNITS) - 1:
                # Last layer - unidirectional for efficiency
                x = tf.keras.layers.GRU(
                    units, return_sequences=True,
                    dropout=self.config.DROPOUT_RATE,
                    recurrent_dropout=self.config.DROPOUT_RATE / 2,
                    name=f'gru_{i+1}'
                )(x)
            else:
                # Bidirectional for better context
                x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(
                        units, return_sequences=True,
                        dropout=self.config.DROPOUT_RATE,
                        recurrent_dropout=self.config.DROPOUT_RATE / 2,
                        name=f'gru_{i+1}'
                    ),
                    name=f'bidirectional_gru_{i+1}'
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
        
        # Output layer for phoneme predictions
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax'),
            name='phoneme_predictions'
        )(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='teacher_phoneme_model')
        
        # Standard optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        
        # Save model architecture
        self.config.MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
        with open(self.config.MODEL_OUTPUT / "model_architecture.json", 'w') as f:
            json.dump(model.to_json(), f, indent=2)
        
        self.logger.info(f"[SUCCESS] Teacher model built with {model.count_params():,} parameters")
        return model
    
    def create_data_pipeline(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create data pipeline optimized for teacher model training"""
        self.logger.info("[PIPELINE] Creating data pipeline...")
        
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
        
        # Optimize training dataset with memory efficiency
        train_ds = (train_ds
                   .shuffle(buffer_size=min(1000, n_train))  # Smaller buffer
                   .batch(self.config.BATCH_SIZE, drop_remainder=True)
                   .prefetch(2))  # Limited prefetch
        
        # Optimize validation dataset
        val_ds = (val_ds
                 .batch(self.config.BATCH_SIZE)
                 .prefetch(2))  # Limited prefetch
        
        # Optimize test dataset
        test_ds = (test_ds
                  .batch(self.config.BATCH_SIZE)
                  .prefetch(2))  # Limited prefetch
        
        return train_ds, val_ds, test_ds
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup standard training callbacks for teacher model"""
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
            EnhancedProgressCallback(verbose=1, update_freq=self.config.PROGRESS_UPDATE_FREQ)
        ]
        
        # Add checkpoint callbacks
        if self.config.SAVE_BEST_ONLY:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.config.MODEL_OUTPUT / "best_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    mode='min',
                    verbose=1
                )
            )
        
        # Periodic checkpoint with custom callback
        if self.config.CHECKPOINT_FREQUENCY > 0:
            def save_checkpoint_callback(epoch, logs):
                if (epoch + 1) % self.config.CHECKPOINT_FREQUENCY == 0:
                    checkpoint_path = self.config.MODEL_OUTPUT / f"checkpoint_epoch_{epoch+1:02d}.keras"
                    self.model.save(str(checkpoint_path))
                    print(f"Epoch {epoch+1}: saving model to {checkpoint_path}")
            
            callbacks.append(
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=save_checkpoint_callback
                )
            )
        
        # Add a callback to print detailed training info
        callbacks.append(
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._log_epoch_summary(epoch, logs)
            )
        )
        
        return callbacks
    
    def _log_epoch_summary(self, epoch: int, logs: Dict):
        """Log detailed epoch summary"""
        if logs:
            summary_parts = [f"Epoch {epoch+1} Summary:"]
            
            # Training metrics
            if 'loss' in logs and 'sparse_categorical_accuracy' in logs:
                summary_parts.append(
                    f"Train -> Loss: {logs['loss']:.4f}, Acc: {logs['sparse_categorical_accuracy']:.4f}"
                )
            
            # Validation metrics
            if 'val_loss' in logs and 'val_sparse_categorical_accuracy' in logs:
                summary_parts.append(
                    f"Val -> Loss: {logs['val_loss']:.4f}, Acc: {logs['val_sparse_categorical_accuracy']:.4f}"
                )
            
            # Learning rate
            if 'lr' in logs:
                summary_parts.append(f"LR: {logs['lr']:.2e}")
            
            self.logger.info(" | ".join(summary_parts))
    
    def train(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        """Train the teacher model with enhanced progress tracking"""
        self.logger.info("[TRAINING] Starting teacher model training...")
        
        # Build model
        self.model = self.build_model(num_classes)
        
        # Create data pipeline
        train_ds, val_ds, test_ds = self.create_data_pipeline(X, y)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Print training configuration
        print(f"\n{'='*80}")
        print(f"[TRAINING CONFIGURATION]")
        print(f"  Total Samples: {len(X):,}")
        print(f"  Batch Size: {self.config.BATCH_SIZE}")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Learning Rate: {self.config.LEARNING_RATE}")
        print(f"  Checkpoint Frequency: Every {self.config.CHECKPOINT_FREQUENCY} epochs")
        print(f"  Progress Updates: Every {self.config.PROGRESS_UPDATE_FREQ} batches")
        print(f"  Model Parameters: {self.model.count_params():,}")
        print(f"{'='*80}")
        
        # Train model
        start_time = time.time()
        
        try:
            # Add memory cleanup before training
            import gc
            gc.collect()
            
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.EPOCHS,
                callbacks=callbacks,
                verbose=1  # Enable Keras built-in progress display
            )
        except (KeyboardInterrupt, SystemExit):
            self.logger.warning("[INTERRUPTED] Training interrupted by user")
            print(f"\n[TRAINING INTERRUPTED] Saving current model state...")
            
            # Save current model state
            interrupted_path = self.config.MODEL_OUTPUT / "interrupted_model.keras"
            self.model.save(str(interrupted_path))
            self.logger.info(f"[SAVE] Interrupted model saved to: {interrupted_path}")
            
            raise
        except Exception as e:
            self.logger.error(f"[ERROR] Training failed: {e}")
            # Save model state on error
            error_path = self.config.MODEL_OUTPUT / "error_model.keras"
            self.model.save(str(error_path))
            self.logger.info(f"[SAVE] Model saved on error to: {error_path}")
            raise
        
        training_time = time.time() - start_time
        self.logger.info(f"[SUCCESS] Training completed in {training_time/3600:.2f} hours")
        
        # Evaluate on test set
        self.logger.info("[EVAL] Evaluating on test set...")
        test_results = self.model.evaluate(test_ds, verbose=1)
        
        # Save final model
        final_model_path = self.config.MODEL_OUTPUT / "teacher_model_final.keras"
        self.model.save(str(final_model_path))
        
        # Also save in TensorFlow SavedModel format (directory)
        tf_model_path = self.config.MODEL_OUTPUT / "teacher_model_final_tf"
        self.model.export(str(tf_model_path))
        
        # Save training artifacts
        self._save_training_artifacts(test_results, training_time)
        
        print(f"\n{'='*80}")
        print(f"[TRAINING COMPLETED SUCCESSFULLY]")
        print(f"  Final Test Loss: {test_results[0]:.4f}")
        print(f"  Final Test Accuracy: {test_results[1]:.4f}")
        print(f"  Training Time: {training_time/3600:.2f} hours")
        print(f"  Model Saved: {final_model_path}")
        print(f"{'='*80}")
        
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

class RealTimeMetricsCallback(tf.keras.callbacks.Callback):
    """Real-time metrics tracking and display"""
    
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.metrics_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Store metrics
            self.metrics_history['epoch'].append(epoch + 1)
            self.metrics_history['loss'].append(logs.get('loss', 0))
            self.metrics_history['accuracy'].append(logs.get('sparse_categorical_accuracy', 0))
            self.metrics_history['val_loss'].append(logs.get('val_loss', 0))
            self.metrics_history['val_accuracy'].append(logs.get('val_sparse_categorical_accuracy', 0))
            self.metrics_history['learning_rate'].append(logs.get('lr', 0))
            
            # Print progress bar style update
            progress = (epoch + 1) / self.params['epochs']
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\rProgress: |{bar}| {progress*100:.1f}% Complete", end='', flush=True)
            
            # Save metrics periodically
            if self.log_dir and (epoch + 1) % 5 == 0:
                self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to CSV file"""
        if self.log_dir:
            metrics_file = Path(self.log_dir) / "realtime_metrics.csv"
            with open(metrics_file, 'w') as f:
                f.write("epoch,loss,accuracy,val_loss,val_accuracy,learning_rate\n")
                for i in range(len(self.metrics_history['epoch'])):
                    f.write(f"{self.metrics_history['epoch'][i]},"
                           f"{self.metrics_history['loss'][i]},"
                           f"{self.metrics_history['accuracy'][i]},"
                           f"{self.metrics_history['val_loss'][i]},"
                           f"{self.metrics_history['val_accuracy'][i]},"
                           f"{self.metrics_history['learning_rate'][i]}\n")

def main():
    """Main training function with enhanced options"""
    parser = argparse.ArgumentParser(description='Train Enhanced Teacher Phoneme Recognition Model')
    parser.add_argument('--data-root', type=str, default='datasets_original/en',
                       help='Root directory containing features_clean and labels_clean')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Initial learning rate')
    parser.add_argument('--segment-length', type=int, default=128,
                       help='Sequence segment length')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                       help='Save checkpoint every N epochs (0 to disable periodic checkpoints)')
    parser.add_argument('--progress-freq', type=int, default=10,
                       help='Update progress every N batches')
    parser.add_argument('--save-best-only', action='store_true', default=True,
                       help='Only save the best model based on validation loss')
    # Enhanced training options
    parser.add_argument('--use-cyclic-lr', action='store_true', default=True,
                       help='Use cyclic learning rate schedule')
    parser.add_argument('--max-lr', type=float, default=2e-3,
                       help='Maximum learning rate for cyclic schedule')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                       help='Minimum learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor (0.0 to disable)')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--architecture', type=str, default='enhanced', 
                       choices=['basic', 'enhanced'], help='Model architecture type')
    # Memory optimization arguments
    parser.add_argument('--max-segments', type=int, default=100000,
                       help='Maximum number of segments to use (for memory efficiency)')
    parser.add_argument('--sample-ratio', type=float, default=0.3,
                       help='Ratio of files to sample for training (0.1-1.0)')
    parser.add_argument('--gpu-memory-limit', type=int, default=16384,
                       help='GPU memory limit in MB (set lower if OOM errors occur)')
    
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
    config.CHECKPOINT_FREQUENCY = args.checkpoint_freq
    config.PROGRESS_UPDATE_FREQ = args.progress_freq
    config.SAVE_BEST_ONLY = args.save_best_only
    # Enhanced options
    config.USE_CYCLIC_LR = args.use_cyclic_lr
    config.MAX_LR = args.max_lr
    config.MIN_LR = args.min_lr
    config.USE_LABEL_SMOOTHING = args.label_smoothing
    config.DROPOUT_RATE = args.dropout_rate
    # Memory optimization settings
    config.MAX_SEGMENTS = args.max_segments
    config.SAMPLE_RATIO = args.sample_ratio
    
    # Adjust architecture based on selection
    if args.architecture == 'basic':
        config.CONV_FILTERS = [64, 128, 128]
        config.GRU_UNITS = [256, 256, 128]
        config.DENSE_UNITS = [128, 64]
    
    # Print configuration
    print(f"\n{'='*70}")
    print("[ENHANCED TRAINING CONFIGURATION]")
    print(f"Architecture: {args.architecture.upper()}")
    print(f"Data Root: {config.DATA_ROOT}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Segment Length: {config.SEGMENT_LENGTH}")
    print(f"Dropout Rate: {config.DROPOUT_RATE}")
    print(f"Label Smoothing: {config.USE_LABEL_SMOOTHING}")
    print(f"Cyclic LR: {config.USE_CYCLIC_LR} (Max: {config.MAX_LR}, Min: {config.MIN_LR})")
    print(f"Checkpoint Frequency: {config.CHECKPOINT_FREQUENCY} epochs")
    print(f"Progress Update Frequency: {config.PROGRESS_UPDATE_FREQ} batches")
    print(f"Save Best Only: {config.SAVE_BEST_ONLY}")
    print(f"Max Segments: {config.MAX_SEGMENTS:,}")
    print(f"Sample Ratio: {config.SAMPLE_RATIO}")
    print(f"{'='*70}")
    
    # Print help message about enhanced features
    if args.checkpoint_freq == 0:
        print("📝 Note: Periodic checkpointing is disabled (--checkpoint-freq=0)")
    else:
        print(f"💾 Note: Checkpoints will be saved every {args.checkpoint_freq} epochs")
    
    print(f"📊 Note: Progress will update every {args.progress_freq} batches")
    print(f"🎯 Note: Best model saving: {'Enabled' if args.save_best_only else 'Disabled'}")
    print(f"🔄 Note: Cyclic learning rate: {'Enabled' if args.use_cyclic_lr else 'Disabled'}")
    print(f"🎛️  Note: Label smoothing: {args.label_smoothing}")
    print(f"🧠 Note: Enhanced architecture with attention and residual connections")
    print(f"⏱️  Note: Real-time metrics will be saved to logs/teacher_training/realtime_metrics.csv")
    print(f"{'='*70}\n")
    
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
