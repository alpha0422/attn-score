// Source code based on pytorch/aten/src/ATen/native/cuda/SoftMax.cu, ce4f311b, 01/2019

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
//#include <ATen/TensorUtils.h>
//#include <ATen/NativeFunctions.h>
//#include <ATen/WrapDimUtils.h>
//#include <THC/THCTensorMathReduce.cuh>
//#include <THC/THCTensorSort.cuh>
//#include <THC/THCThrustAllocator.cuh>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>
//#include <type_traits>

#include <THC/THC.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>

using Tensor = at::Tensor;
using TensorList = at::TensorList;
using ScalarType = at::ScalarType;
using at::acc_type;

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : logsum(max_input + std::log(sum)) {}

  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_log_sum_exp)
    : logsum(max_log_sum_exp) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(input - logsum);
}

  const AccumT logsum;
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - std::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input)
    , sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(std::exp(input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxBackwardEpilogue {
  __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - output * sum);
  }

  const AccumT sum;
};

const int max_threads = 1024;

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));
  while (block_size < max_block_size) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(32));
  return dim3(block_size);
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};



////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP) {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      threadVal = r(threadVal, tmp[j]);
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(outscalar_t *output, scalar_t *input, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      input, classes, MaxFloat<scalar_t, accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = input[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
  }

  for (; offset < classes; offset += blockDim.x)
    output[offset] = epilogue(input[offset]);
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxBackward(scalar_t *gradInput, outscalar_t *output, outscalar_t *gradOutput, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  accscalar_t threadSum = ilpReduce<AddFloat, 4, outscalar_t, accscalar_t>(
      gradOutput, classes, AddFloat<outscalar_t, accscalar_t>(), accscalar_t(0));
  accscalar_t sum_k = blockReduce<Add, accscalar_t>(
        sdata, threadSum, Add<accscalar_t>(), accscalar_t(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum_k);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    outscalar_t tmpGradOutput[ILP];
    outscalar_t tmpOutput[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
      tmpOutput[j] = output[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      gradInput[offset + j * blockDim.x] = epilogue(tmpGradOutput[j], tmpOutput[j]);
  }

  for (; offset < classes; offset += blockDim.x)
    gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
}






template<template<typename, typename, typename> class Epilogue>
Tensor host_softmax(const Tensor & input_, const int64_t dim_, const bool half_to_float){
  if (half_to_float) AT_ASSERTM(input_.type().scalarType() == ScalarType::Half,"conversion is supported for Half type only");
  auto input = input_.contiguous();
  Tensor output = half_to_float ? at::empty_like(input, input.options().dtype(ScalarType::Float)) : at::empty_like(input);
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = at::maybe_wrap_dim(dim_, input.dim());
  AT_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // This kernel spawns a block per each element in the batch.
    // XXX: it assumes that inner_size == 1
    if (inner_size == 1) {
      const int ILP = 2;
      dim3 grid(outer_size);
      dim3 block = SoftMax_getBlockSize(ILP, dim_size);
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "host_softmax", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      if (!half_to_float) {
          cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
            <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
              output.data<scalar_t>(), input.data<scalar_t>(), dim_size
          );
      } else {
          cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
            <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
              output.data<accscalar_t>(), input.data<scalar_t>(), dim_size
          );
      }
      });
    // This kernel runs in a 2D grid, where each application along y dimension has a fixed
    // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
    // Reductions over dim are done in a single-threaded manner.
    }
    THCudaCheck(cudaGetLastError());
  }
  return output;
}

template<template<typename, typename, typename> class Epilogue>
Tensor host_softmax_backward(const Tensor &grad_, const Tensor &output_, int64_t dim_, bool half_to_float){
  int64_t dim = at::maybe_wrap_dim(dim_, grad_.dim());
  Tensor gI = half_to_float ? at::empty_like(grad_, grad_.options().dtype(ScalarType::Half)) : at::empty_like(grad_);
  if (grad_.numel() == 0) {
    return gI;
  }
  auto grad = grad_.contiguous();
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (grad.dim() == 0) grad = grad.view(1);
  AT_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");
  auto output = output_.contiguous();
  if (output.dim() == 0) output = output.view(1);
  int64_t outer_size = 1;
  int64_t dim_size = output.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
  for (int64_t i = dim + 1; i < output.dim(); ++i)
    inner_size *= output.size(i);
// See descriptions of kernels above.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (inner_size == 1) {
    const int ILP = 2;
    dim3 grid(outer_size);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gI.type(), "host_softmax_backward", [&] {
    using accscalar_t = acc_type<scalar_t, true>;
    if (!half_to_float) {
        cunn_SoftMaxBackward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
         <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
            gI.data<scalar_t>(), output.data<scalar_t>(), grad.data<scalar_t>(), dim_size
    );
    } else {
        cunn_SoftMaxBackward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
         <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
            gI.data<scalar_t>(), output.data<accscalar_t>(), grad.data<accscalar_t>(), dim_size
    );
    }
    });
  }
  THCudaCheck(cudaGetLastError());
  return gI;
}

at::Tensor attn_score_forward_cuda(
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn,
    const uint32_t tile_x,
    const uint32_t tile_y) {

	return attn_query;
}

std::vector<at::Tensor> attn_score_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &linear_attn,
    const uint32_t tile_x,
    const uint32_t tile_y) {

	std::vector<at::Tensor> ret = {grad_output};
	return ret;	
}

