import os
import sys

# Set CUDA environment variables
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if os.path.exists(cuda_path):
    os.environ["CUDA_HOME"] = cuda_path
    os.environ["CUDA_PATH"] = cuda_path
    
    # Add CUDA bin directory to PATH
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.exists(cuda_bin):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    
    # Add CUDA lib directory to PATH
    cuda_lib = os.path.join(cuda_path, "lib", "x64")
    if os.path.exists(cuda_lib):
        os.environ["PATH"] = cuda_lib + os.pathsep + os.environ["PATH"]
    else:
        # Try alternative paths
        alt_lib = os.path.join(cuda_path, "lib")
        if os.path.exists(alt_lib):
            os.environ["PATH"] = alt_lib + os.pathsep + os.environ["PATH"]
    
    print(f"CUDA environment variables set to {cuda_path}")
else:
    print(f"Warning: CUDA path {cuda_path} not found")

# Import torch after setting environment variables
import torch

print("\n=== PyTorch CUDA Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    
    # Test CUDA with a simple operation
    print("\n=== Testing CUDA with tensor operations ===")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"Test tensor device: {z.device}")
    print("CUDA test successful!")
else:
    print("\nCUDA is not available. Here are some troubleshooting steps:")
    print("1. Check if you have an NVIDIA GPU")
    print("2. Make sure you have the latest NVIDIA drivers installed")
    print("3. Verify that CUDA toolkit is installed correctly")
    print("4. Check if PyTorch was installed with CUDA support")
    print("5. Try reinstalling PyTorch with CUDA support using:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126") 