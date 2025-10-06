# Build once (run this script separately)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="hash_encoder",
    ext_modules=[
        CUDAExtension(
            name="hash_encoder",
            sources=["src/hashencoder.cu", "src/bindings.cpp"],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '-allow-unsupported-compiler',
                        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_HALF2_OPERATORS__']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
