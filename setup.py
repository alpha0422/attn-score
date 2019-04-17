from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attn_score',
    version='0.1',
    description='Higher performance attention score calculation.',
    packages=find_packages(), 
    ext_modules=[
        CUDAExtension('attn_score_cuda', [
            'csrc/attn_score_cuda.cpp',
            'csrc/attn_score_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O2',],
            'nvcc':['--gpu-architecture=sm_70',]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    test_suite="tests",
)
