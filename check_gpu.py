#!/usr/bin/env python3
"""
GPU Detection and TensorFlow Configuration Test
"""
import tensorflow as tf
import os

def check_gpu_setup():
    """Check GPU availability and configuration"""
    print("=" * 60)
    print("TENSORFLOW GPU CONFIGURATION CHECK")
    print("=" * 60)
    
    # TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if CUDA is built with TensorFlow
    print(f"CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}")
    
    # List physical devices
    print("\nPhysical devices:")
    for device in tf.config.list_physical_devices():
        print(f"  - {device}")
    
    # GPU specific checks
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices found: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                # Get GPU details
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {gpu_details}")
            except Exception as e:
                print(f"    Could not get details: {e}")
    else:
        print("  No GPU devices found!")
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. CUDA drivers not installed")
        print("  3. TensorFlow CPU-only version installed")
        print("  4. GPU not compatible with TensorFlow")
    
    # Test GPU availability
    print(f"\nGPU available for TensorFlow: {tf.test.is_gpu_available()}")
    
    # Environment variables
    print("\nRelevant environment variables:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_GPU_ALLOCATOR']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Test simple GPU operation
    if gpus:
        print("\nTesting GPU operation...")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
                print(f"  GPU matrix multiplication successful: {c.numpy()}")
        except Exception as e:
            print(f"  GPU operation failed: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_gpu_setup()
