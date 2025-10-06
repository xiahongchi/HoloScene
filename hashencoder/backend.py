from distutils.command.build import build
import os
from torch.utils.cpp_extension import load
from pathlib import Path

_src_path = os.path.dirname(os.path.abspath(__file__))

# Create build directory inside hashencoder folder
_build_dir = os.path.join(_src_path, 'build')
Path(_build_dir).mkdir(parents=True, exist_ok=True)

_backend = load(name='_hash_encoder',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=[
                    '-O3', '-std=c++17', '-allow-unsupported-compiler',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]],
                build_directory=_build_dir,
                verbose=True,
                )

__all__ = ['_backend']