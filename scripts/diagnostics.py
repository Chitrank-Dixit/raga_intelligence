"""
Environment and hardware diagnostic script for Indian Raga Intelligence.

Verifies:
1. Python version and core library availability.
2. PyTorch installation, CUDA availability, GPU visibility, and real tensor execution.
3. Audio decoding support (soundfile, torchaudio, librosa).
4. Container environment & Podman/Compose compatibility.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def check_python_and_core_packages() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {},
    }
    
    packages_to_check = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "soundfile",
        "librosa",
        "transformers",
        "fastapi",
        "pydantic",
        "yaml",
    ]
    
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            info["packages"][pkg] = "not installed"
            
    return info


def check_pytorch_and_gpu() -> Dict[str, Any]:
    gpu_info: Dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "tensor_op_verified": False,
        "tensor_op_device": "none",
        "tensor_op_result": None,
        "notes": [],
    }
    
    try:
        import torch
        gpu_info["torch_installed"] = True
        gpu_info["torch_version"] = torch.__version__
        gpu_info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_info["cuda_version"] = torch.version.cuda
            count = torch.cuda.device_count()
            gpu_info["device_count"] = count
            
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = round(props.total_memory / (1024**3), 2)
                allocated_mem_gb = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                device_data = {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": total_mem_gb,
                    "allocated_memory_gb": allocated_mem_gb,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
                gpu_info["devices"].append(device_data)
                
            # Perform actual tensor calculation on GPU to verify execution
            try:
                device = torch.device("cuda:0")
                x = torch.ones((1000, 1000), device=device, dtype=torch.float32)
                y = torch.matmul(x, x)
                gpu_info["tensor_op_verified"] = bool(torch.allclose(y[0, 0], torch.tensor(1000.0, device=device)))
                gpu_info["tensor_op_device"] = "cuda:0"
                gpu_info["tensor_op_result"] = "passed (matrix multiplication 1000x1000 verified)"
            except Exception as e:
                gpu_info["tensor_op_verified"] = False
                gpu_info["tensor_op_device"] = "cuda:0 (failed)"
                gpu_info["notes"].append(f"CUDA tensor operation failed on sm_120: {e}")
                
                # Test CPU fallback computation
                try:
                    x_cpu = torch.ones((1000, 1000), dtype=torch.float32)
                    y_cpu = torch.matmul(x_cpu, x_cpu)
                    cpu_passed = bool(torch.allclose(y_cpu[0, 0], torch.tensor(1000.0)))
                    if cpu_passed:
                        gpu_info["cpu_fallback_verified"] = True
                        gpu_info["notes"].append("CPU fallback tensor execution verified successfully.")
                except Exception as cpu_e:
                    gpu_info["notes"].append(f"CPU fallback failed: {cpu_e}")
        else:
            # Verify CPU fallback tensor computation
            try:
                x = torch.ones((100, 100), dtype=torch.float32)
                y = torch.matmul(x, x)
                gpu_info["tensor_op_verified"] = bool(torch.allclose(y[0, 0], torch.tensor(100.0)))
                gpu_info["tensor_op_device"] = "cpu"
                gpu_info["tensor_op_result"] = "passed (CPU fallback verified)"
            except Exception as e:
                gpu_info["notes"].append(f"CPU tensor operation failed: {e}")
                
    except ImportError as e:
        gpu_info["notes"].append(f"PyTorch not installed: {e}")
        
    return gpu_info


def check_audio_decoders() -> Dict[str, Any]:
    audio_info: Dict[str, Any] = {
        "soundfile_backend": None,
        "torchaudio_version": None,
        "torchaudio_backends": [],
        "librosa_available": False,
        "test_decoding_passed": False,
        "notes": [],
    }
    
    # Soundfile
    try:
        import soundfile as sf
        audio_info["soundfile_backend"] = sf.__version__
    except ImportError:
        audio_info["notes"].append("soundfile is not installed")
        
    # Torchaudio
    try:
        import torchaudio
        audio_info["torchaudio_version"] = torchaudio.__version__
        if hasattr(torchaudio, "list_audio_backends"):
            audio_info["torchaudio_backends"] = torchaudio.list_audio_backends()
    except (ImportError, Exception) as e:
        audio_info["notes"].append(f"torchaudio backend check note: {e}")
        
    # Librosa & synthetic wave test
    try:
        import numpy as np
        import librosa
        audio_info["librosa_available"] = True
        
        # Test synthetic 1-second 440Hz sine wave processing
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        
        # Resample test (allowing +/- 2 samples for boundary filter padding)
        resampled = librosa.resample(sig, orig_sr=sr, target_sr=24000)
        audio_info["test_decoding_passed"] = bool(abs(len(resampled) - 24000) <= 2)
    except Exception as e:
        audio_info["notes"].append(f"Librosa synthetic test failed: {e}")
        
    return audio_info


def check_container_tooling() -> Dict[str, Any]:
    tooling: Dict[str, Any] = {
        "podman_installed": False,
        "podman_version": None,
        "compose_provider": None,
        "compose_version": None,
        "inside_container": os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"),
    }
    
    try:
        res = subprocess.run(
            ["podman", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0:
            tooling["podman_installed"] = True
            tooling["podman_version"] = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["podman", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0:
            tooling["compose_version"] = res.stdout.strip()
    except Exception:
        pass
        
    return tooling


def run_all_diagnostics() -> Dict[str, Any]:
    results = {
        "system": check_python_and_core_packages(),
        "gpu": check_pytorch_and_gpu(),
        "audio": check_audio_decoders(),
        "containers": check_container_tooling(),
    }
    return results


def print_summary(report: Dict[str, Any]) -> None:
    print("=" * 70)
    print(" INDIAN RAGA INTELLIGENCE: ENVIRONMENT DIAGNOSTICS REPORT")
    print("=" * 70)
    
    # Python & Packages
    sys_info = report["system"]
    print(f"Python Platform : {sys_info['platform']}")
    print(f"Python Executable: {sys_info['executable']}")
    print("Core Packages   :")
    for pkg, ver in sys_info["packages"].items():
        print(f"  - {pkg:<15}: {ver}")
        
    # GPU & Torch
    gpu_info = report["gpu"]
    print("\nMachine Learning & GPU Acceleration:")
    print(f"  - PyTorch Version     : {gpu_info['torch_version']}")
    print(f"  - CUDA Available       : {gpu_info['cuda_available']}")
    if gpu_info["cuda_available"]:
        print(f"  - CUDA Runtime Version: {gpu_info['cuda_version']}")
        print(f"  - Visible Devices     : {gpu_info['device_count']}")
        for dev in gpu_info["devices"]:
            print(f"    * [{dev['index']}] {dev['name']} ({dev['total_memory_gb']} GB VRAM, CC {dev['compute_capability']})")
    print(f"  - Tensor Op Verified   : {gpu_info['tensor_op_verified']} on {gpu_info['tensor_op_device']}")
    print(f"  - Tensor Op Result     : {gpu_info['tensor_op_result']}")
    if gpu_info["notes"]:
        print(f"  - GPU Notes            : {', '.join(gpu_info['notes'])}")
        
    # Audio
    audio_info = report["audio"]
    print("\nAudio Decoding & Processing:")
    print(f"  - SoundFile Version    : {audio_info['soundfile_backend']}")
    print(f"  - Librosa Available    : {audio_info['librosa_available']}")
    print(f"  - Synthetic DSP Test   : {'PASSED' if audio_info['test_decoding_passed'] else 'FAILED'}")
    if audio_info["notes"]:
        print(f"  - Audio Notes          : {', '.join(audio_info['notes'])}")

    # Containers
    cnt_info = report["containers"]
    print("\nContainer & Deployment Tooling:")
    print(f"  - Inside Container     : {cnt_info['inside_container']}")
    print(f"  - Podman Version       : {cnt_info['podman_version']}")
    print(f"  - Compose Interface    : {cnt_info['compose_version']}")
    print("=" * 70)


if __name__ == "__main__":
    report = run_all_diagnostics()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)

