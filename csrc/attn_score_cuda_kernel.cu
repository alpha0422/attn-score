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

#if 0
#define DPRINTF(fmt, args...)                       \
    do{                                             \
        fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt,  \
            __FILE__, __LINE__, __func__, ##args);  \
    } while(false)
#else
#define DPRINTF(fmt, args...) do{ } while (false)
#endif

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
  uint32_t mask = (((uint64_t)1) << (blockDim.x / 32)) - 1;
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      __syncwarp(mask);
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

/** Each block process TILE_Q*TILE_K*hidden volumn. */
template <int TILE, typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void
cunn_AttnScoreForward(
    outscalar_t *output,
    const scalar_t* __restrict__ attn_query,
    const scalar_t* __restrict__ attn_keys,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ linear_attn,
    int t_q,
    int t_k,
    int hidden) {
    
    extern __shared__ unsigned char smem[];
    auto tmp_q = reinterpret_cast<scalar_t*>(smem);
    auto tmp_k = tmp_q + TILE * blockDim.x;
    auto tmp_b = tmp_k + TILE * blockDim.x;
    auto tmp_l = tmp_b + blockDim.x;
    auto tmp_o = reinterpret_cast<accscalar_t*>(tmp_l + blockDim.x);

    int batch_id = blockIdx.x;
    int q_start = blockIdx.y * TILE;
    int k_start = blockIdx.z * TILE;
    
    attn_query += batch_id*t_q*hidden + q_start*hidden;
    attn_keys += batch_id*t_k*hidden + k_start*hidden;
    output += batch_id*t_q*t_k;

    // initialize intermediate result
    #pragma unroll
    for (int i = 0; i < TILE; i++)
        #pragma unroll
        for (int j = 0; j < TILE; j++)
            tmp_o[i*TILE*blockDim.x+j*blockDim.x+threadIdx.x] = 0;

    // ilpReduce
    int offset = threadIdx.x;
    int last = hidden % blockDim.x;

    // ilpReduce on regular data
    for (; offset < hidden - last; offset += blockDim.x) {
        // prolog: load query slices to shared memory
        for (int i = 0; i < t_q - q_start && i < TILE; i++)
            tmp_q[i*blockDim.x+threadIdx.x] = attn_query[i*hidden+offset];

        // prolog: load key slices to shared memory
        for (int i = 0; i < t_k - k_start && i < TILE; i++)
            tmp_k[i*blockDim.x+threadIdx.x] = attn_keys[i*hidden+offset];

        // prolog: load bias and linear_attn slices to shared memory
        tmp_b[threadIdx.x] = bias[offset];
        tmp_l[threadIdx.x] = linear_attn[offset];

        // main loop
        for (int i = 0; i < t_q - q_start && i < TILE; i++) {
            for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                accscalar_t s = static_cast<accscalar_t>(
                    tmp_q[i*blockDim.x+threadIdx.x] +
                    tmp_k[j*blockDim.x+threadIdx.x] +
                    tmp_b[threadIdx.x]);
                tmp_o[i*TILE*blockDim.x+j*blockDim.x+threadIdx.x] += tanhf(s) * tmp_l[threadIdx.x];
                DPRINTF("threadVal: %d %d %f\n", i, j, tanhf(s) * tmp_l[threadIdx.x]);
            }
        }
    }

    // ilpReduce on boundary
    for (; offset < hidden; offset += blockDim.x) {
        // prolog: load query slices to shared memory
        for (int i = 0; i < t_q - q_start && i < TILE; i++)
            tmp_q[i*blockDim.x+threadIdx.x] = attn_query[i*hidden+offset];

        // prolog: load key slices to shared memory
        for (int i = 0; i < t_k - k_start && i < TILE; i++)
            tmp_k[i*blockDim.x+threadIdx.x] = attn_keys[i*hidden+offset];

        // prolog: load bias and linear_attn slices to shared memory
        tmp_b[threadIdx.x] = bias[offset];
        tmp_l[threadIdx.x] = linear_attn[offset];

        // main loop
        for (int i = 0; i < t_q - q_start && i < TILE; i++) {
            for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                accscalar_t s = static_cast<accscalar_t>(
                    tmp_q[i*blockDim.x+threadIdx.x] +
                    tmp_k[j*blockDim.x+threadIdx.x] +
                    tmp_b[threadIdx.x]);
                tmp_o[i*TILE*blockDim.x+j*blockDim.x+threadIdx.x] += tanhf(s) * tmp_l[threadIdx.x];
            }
        }
    }

    // blockReduce
    __syncthreads();

    // First warp will perform per-warp reductions for the remaining warps
    uint32_t mask = (((uint64_t)1) << (blockDim.x / 32)) - 1;
    if (threadIdx.x < 32) {
        int lane = threadIdx.x % 32;
        if (lane < blockDim.x / 32) {
            for (int i = 0; i < t_q - q_start && i < TILE; i++) {
                for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                    accscalar_t warpVal = static_cast<accscalar_t>(0);
                    #pragma unroll
                    for (int k = 0; k < 32; ++k) {
                        DPRINTF("warpVal: %d %d %d %d %f %f\n", lane, i, j, k, tmp_o[i*TILE*blockDim.x+j*blockDim.x+lane*32+k], warpVal);
                        warpVal += tmp_o[i*TILE*blockDim.x+j*blockDim.x+lane*32+k];
                    }
                    __syncwarp(mask);
                    tmp_o[i*TILE*blockDim.x+j*blockDim.x+lane] = warpVal;
                    DPRINTF("warpVal: %d %d %d %f\n", lane, i, j, warpVal);
                }
            }
        }
    }

    __syncthreads();

    // First thread will perform a reduction of the above per-warp reductions
    if (threadIdx.x == 0) {
        for (int i = 0; i < t_q - q_start && i < TILE; i++) {
            for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                accscalar_t blockVal = static_cast<accscalar_t>(0);
                for (int k = 0; k < blockDim.x / 32; ++k) {
                    blockVal += tmp_o[i*TILE*blockDim.x+j*blockDim.x+k];
                }
                output[(i+q_start)*t_k+(j+k_start)] = static_cast<outscalar_t>(blockVal);
                DPRINTF("blockVal: %d %d %f\n", i, j, blockVal);
            }
        }
    }

    // Sync and broadcast
    __syncthreads();
}

at::Tensor attn_score_forward_cuda(
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn) {
    int batch_sz = attn_query.size(0);
    int t_q = attn_query.size(1);
    int t_k = attn_keys.size(1);
    int hidden = attn_query.size(2);

    Tensor output = at::empty({batch_sz, t_q, t_k}, attn_query.options());

    const int TILE = 4;
    int grid_x = batch_sz;
    int grid_y = (t_q + TILE - 1) / TILE;
    int grid_z = (t_k + TILE - 1) / TILE;

    // Each block process TILE_Q*TILE_K*hidden volumn. 
    dim3 block(128);
    dim3 grid(grid_x, grid_y, grid_z);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Each block load (TILE_Q+TILE_K)*block.x volumn each time
    // Each block load block.x volumn bias and linear_attn
    // Each thread reserve its local results for intra block reduction
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_query.type(), "attn_score_fprop", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        cunn_AttnScoreForward<TILE, scalar_t, accscalar_t, scalar_t>
        <<<grid, block, (2*TILE+2)*block.x * sizeof(scalar_t)+
            block.x * TILE * TILE * sizeof(accscalar_t), stream>>>(
            output.data<scalar_t>(), attn_query.data<scalar_t>(),
            attn_keys.data<scalar_t>(), bias.data<scalar_t>(),
            linear_attn.data<scalar_t>(), t_q, t_k, hidden
        );
    });

    THCudaCheck(cudaGetLastError());
	return output;
}

/**
 * Each block process batch_sz*t_q*t_k*ILP volumn.
 * Each thread process TILE*TILE*ILP volumn a time.
 */
template <int LEN, int TILE, int BZ, typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void
cunn_AttnScoreBackward(
    outscalar_t *grad_query,
    outscalar_t *grad_keys,
    outscalar_t *grad_bias,
    outscalar_t *grad_lin,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ attn_query,
    const scalar_t* __restrict__ attn_keys,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ linear_attn,
    int batch_sz,
    int t_q,
    int t_k,
    int hidden) {

    extern __shared__ unsigned char smem[];
    auto tmp_qk = reinterpret_cast<accscalar_t*>(smem);
    auto tmp_it = tmp_qk + TILE * TILE * LEN;
    auto tmp_gq = tmp_it + TILE * TILE * LEN;
    auto tmp_gk = tmp_gq + t_q * LEN;
    auto tmp_gb = tmp_gk + t_k * LEN;
    auto tmp_gl = tmp_gb + blockDim.x;
    auto tmp_kk = tmp_gl + blockDim.x;
    auto tmp_qq = tmp_kk + blockDim.x;
    auto tmp_bb = tmp_qq + blockDim.x;
    auto tmp_ll = tmp_bb + blockDim.x;
    auto tmp_q = reinterpret_cast<scalar_t*>(tmp_ll + blockDim.x);
    auto tmp_k = tmp_q + t_q * LEN;
    auto tmp_b = tmp_k + t_k * LEN;
    auto tmp_l = tmp_b + LEN;

    // final reduce to LEN
    assert(LEN < blockDim.x);
    // reduce 3D to 2D, assume one thread reduce same dim
    assert(blockDim.x % (TILE * LEN) == 0);
    assert((TILE * TILE * LEN) % blockDim.x == 0);

    // initialize gradients to zero
    tmp_gl[threadIdx.x] = 0;
    tmp_gb[threadIdx.x] = 0;

    // update pointer to hidden volume offset
    bias += blockIdx.x * LEN;
    linear_attn += blockIdx.x * LEN;

    // load bias volume to shared memory
    for (int i=threadIdx.x; i<LEN; i+=blockDim.x) {
        tmp_b[i] = bias[i];
        tmp_l[i] = linear_attn[i];
    }

    for (int n=0; n<batch_sz; n++) {
        // initialize gradients to zero
        for (int i=threadIdx.x; i<t_q*LEN; i+=blockDim.x)
            tmp_gq[i] = 0;
        for (int i=threadIdx.x; i<t_k*LEN; i+=blockDim.x)
            tmp_gk[i] = 0;

        // load batch specific data to shared memory
        for (int i=threadIdx.x; i<t_q*LEN; i+=blockDim.x)
            tmp_q[i] = attn_query[i/LEN*hidden + blockIdx.x*LEN+i%LEN];
        for (int i=threadIdx.x; i<t_k*LEN; i+=blockDim.x)
            tmp_k[i] = attn_keys[i/LEN*hidden + blockIdx.x*LEN+i%LEN];

        __syncthreads();

        // loop on each tile
        for (int i=0; i<t_q; i+=TILE) {
            for (int j=0; j<t_k; j+=TILE) {
                // main loop
                for (int k=threadIdx.x; k<TILE*TILE*LEN; k+=blockDim.x) {
                    int h_id = k % LEN;
                    int k_id = k / LEN % TILE;
                    int q_id = k / TILE / LEN;

                    accscalar_t s = 0, t = 0;
                    // re-compute fprop intermediate result
                    if (k_id + j < t_k && q_id + i < t_q) {
                        scalar_t go = grad_output[(q_id+i)*t_k+k_id+j];
                        t = static_cast<scalar_t>(tmp_q[(i + q_id) * LEN + h_id] +
                            tmp_k[(j + k_id) * LEN + h_id] + tmp_b[h_id]);
                        t = static_cast<scalar_t>(tanhf(t));
                        s = t * go;
                        t = static_cast<scalar_t>(tmp_l[h_id] * go * (1.f - t * t));
                    }
                    tmp_qk[k] = s;
                    tmp_it[k] = t;
                }

                __syncthreads();

                // reduction on query and key gradients
                accscalar_t g_q = 0, g_k = 0, g_l = 0;
                for (int k=threadIdx.x; k<TILE*TILE*LEN; k+=blockDim.x) {
                    int h_id = k % LEN;
                    int k_id = k / LEN % TILE;
                    int q_id = k / (LEN * TILE);

                    g_q += tmp_it[k_id*TILE*LEN+q_id*LEN+h_id];
                    g_k += tmp_it[k];
                    g_l += tmp_qk[k];
                }
                tmp_qq[threadIdx.x] = g_q;
                tmp_kk[threadIdx.x] = g_k;
                tmp_ll[threadIdx.x] = g_l;
                __syncthreads();

                for (int s=blockDim.x/2; s>=TILE*LEN; s>>=1) {
                    if (threadIdx.x < s) {
                        tmp_qq[threadIdx.x] += tmp_qq[threadIdx.x + s];
                        tmp_kk[threadIdx.x] += tmp_kk[threadIdx.x + s];
                        tmp_ll[threadIdx.x] += tmp_ll[threadIdx.x + s];
                    }
                    __syncthreads();
                }

                int h_id = threadIdx.x % LEN;
                int qkid = threadIdx.x / LEN;
                if (i + qkid < t_q && threadIdx.x < TILE * LEN)
                    tmp_gq[(i+qkid)*LEN+h_id] += tmp_qq[threadIdx.x];
                if (j + qkid < t_k && threadIdx.x < TILE * LEN)
                    tmp_gk[(j+qkid)*LEN+h_id] += tmp_kk[threadIdx.x];
                if (threadIdx.x < TILE * LEN)
                    tmp_gb[threadIdx.x] += tmp_qq[threadIdx.x];
                if (threadIdx.x < TILE * LEN)
                    tmp_gl[threadIdx.x] += tmp_ll[threadIdx.x];
            }
        }

        __syncthreads();

        // write query and keys gradients
        for (int i=threadIdx.x; i<t_q*LEN; i+=blockDim.x) {
            int q_id = i / LEN;
            int h_id = i % LEN;
            grad_query[n*t_q*hidden+q_id*hidden+blockIdx.x*LEN+h_id] = tmp_gq[i];
        }
        for (int i=threadIdx.x; i<t_k*LEN; i+=blockDim.x) {
            int k_id = i / LEN;
            int h_id = i % LEN;
            grad_keys[n*t_k*hidden+k_id*hidden+blockIdx.x*LEN+h_id] = tmp_gk[i];
        }

        // update pointer for next batch
        grad_output += t_q * t_k;
        attn_query += t_q * hidden;
        attn_keys += t_k * hidden;
    }
    
    __syncthreads();

    // reduction bias and linear_attn inside one block
    unsigned int tid = threadIdx.x;
    if (blockDim.x >= 1024 && tid < 512) {
        tmp_gl[tid] += tmp_gl[tid + 512];
        tmp_gb[tid] += tmp_gb[tid + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        tmp_gl[tid] += tmp_gl[tid + 256];
        tmp_gb[tid] += tmp_gb[tid + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        tmp_gl[tid] += tmp_gl[tid + 128];
        tmp_gb[tid] += tmp_gb[tid + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        tmp_gl[tid] += tmp_gl[tid + 64];
        tmp_gb[tid] += tmp_gb[tid + 64];
    }
    __syncthreads();
    static_assert(LEN < 32, "LEN too large.");
    if (tid < 32) {
        accscalar_t tl, tb;
        #pragma unroll
        for (int m=32; m>=LEN; m>>=1) {
            tl = tmp_gl[tid] + tmp_gl[tid + m];
            tb = tmp_gb[tid] + tmp_gb[tid + m];
            __syncwarp();
            tmp_gl[tid] = tl;
            tmp_gb[tid] = tb;
        }
    }
    __syncthreads();
    if (tid < LEN) {
        grad_lin[blockIdx.x*LEN+tid] = tmp_gl[tid];
        grad_bias[blockIdx.x*LEN+tid] = tmp_gb[tid];
    }
}

std::vector<at::Tensor> attn_score_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn) {

    int batch_sz = attn_query.size(0);
    int t_q = attn_query.size(1);
    int t_k = attn_keys.size(1);
    int hidden = attn_query.size(2);

    Tensor grad_query = at::empty_like(attn_query);
    Tensor grad_keys = at::empty_like(attn_keys);
    Tensor grad_bias = at::empty_like(bias);
    Tensor grad_lin = at::empty_like(linear_attn);

    const int BZ = 8;
    const int TILE = 16;
    const int LEN = 4;

    dim3 block(128);
    dim3 grid((hidden+LEN-1)/LEN);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_query.type(), "attn_score_bprop", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        cunn_AttnScoreBackward<LEN, TILE, BZ, scalar_t, accscalar_t, scalar_t>
        <<<grid, block, ((2*TILE*TILE + t_q + t_k) * LEN + 6 * block.x) * sizeof(accscalar_t) +
            (t_q + t_k + 2) * LEN * sizeof(scalar_t) , stream>>>(
            grad_query.data<scalar_t>(), grad_keys.data<scalar_t>(),
            grad_bias.data<scalar_t>(), grad_lin.data<scalar_t>(),
            grad_output.data<scalar_t>(), attn_query.data<scalar_t>(),
            attn_keys.data<scalar_t>(), bias.data<scalar_t>(),
            linear_attn.data<scalar_t>(), batch_sz, t_q, t_k, hidden
        );
    });

    THCudaCheck(cudaGetLastError());
	std::vector<at::Tensor> ret = {grad_query, grad_keys, grad_bias, grad_lin};
	return ret;	
}

