from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='parasin_cuda_kernel',
    ext_modules=[
        CUDAExtension(
            name='parasin_cuda_kernel',
            sources=['parasin_cuda_kernel.cu'],
            include_dirs=[
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include',
                'C:/Users/Amirhossein/anaconda3/Lib/site-packages/torch/include',
                'C:/Users/Amirhossein/anaconda3/Lib/site-packages/torch/include/torch/csrc/api/include',
            ],
            library_dirs=[
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64'
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-gencode=arch=compute_86,code=sm_86',  # Adjust for your GPU architecture
                    '--expt-relaxed-constexpr'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
