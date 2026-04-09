Assignment 1 leaderboard [[Leaderboard]](https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard)

Assignment 2 is out [[A2]](https://github.com/stanford-cs336/spring2025-assignment2-systems)

Last lecture: high-level overview of GPUs and performance

This lecture: benchmarking/profiling + write kernels

## Hardware

<img src="../../materials/spring2025-lectures/var/files/image-672bd77c57df485d07926615162a44d5-https_miro_medium_com_v2_resize_fit_2000_format_webp_1_6xoBKi5kL2dZpivFe1-zgw_jpeg" width="800">

Compute: streaming multiprocessors (SMs) [A100: 108]

Memory:

- DRAM [A100: 80GB] - big, slow

- L2 cache [A100: 40MB]

- L1 cache [A100: 192KB per SM] - small, fast

You can look at the specs on your actual GPU.

8 devices

0: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=a7bfde5d-8151-8a6d-fdfc-b31d020cd2f5, L2_cache_size=50MB)

1: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=049a2adb-0276-9a0e-0efe-c5736c4cc09a, L2_cache_size=50MB)

2: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=d717d9a2-e829-e5ad-bf0b-c1d6c454b8e0, L2_cache_size=50MB)

3: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=2095bca5-683d-0952-bd46-659165c38a97, L2_cache_size=50MB)

4: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=c1e633f3-bddf-17df-0c6f-ac98f426df09, L2_cache_size=50MB)

5: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=a061862e-4d4d-ae4b-f60f-4d5d499b856c, L2_cache_size=50MB)

6: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=9cceb524-8edb-15d9-f620-2f4f029aeb45, L2_cache_size=50MB)

7: _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81090MB, multi_processor_count=132, uuid=62f395b0-f63d-2a9d-d202-53f798ada4f4, L2_cache_size=50MB)

Basic structure: run f(i) for all i = 0, ..., N-1

## Execution model

<img src="../../materials/spring2025-lectures/var/files/image-1390e250b108c0d04b315ac7995eccd0-https_docs_nvidia_com_cuda_parallel-thread-execution__images_grid-with-CTAs_png" width="600">

- *Thread*: process individual index (i.e., f(i))

- *Thread block* (a.k.a. concurrent thread arrays): scheduled on a single SM

- *Grid*: collection of thread blocks

Why thread blocks? Shared memory.

- Intuition: group f(i)'s that read similar data together

- Threads within a thread block have shared memory (as fast as L1 cache) [A100: 164KB]

- Can synchronize threads (for reading/writing) within a block (but not across blocks)

### Hardware and execution interact.

<img src="../../materials/spring2025-lectures/var/files/image-f256f44a88048865b9ad6afcedde912b-https_developer-blogs_nvidia_com_wp-content_uploads_2019_06_pasted-image-0_png" width="400">

Thread blocks scheduled onto SMs in waves.

Problem: last wave has fewer thread blocks, leaving some SMs idle (low occupancy).

Wave quantization: make number of thread blocks divide # SMs.

Rule of thumb: number of thread blocks should be >= 4x # SMs

Challenge: some aspects of hardware are hidden from the execution model (e.g., scheduling, # SMs).

### Arithmetic intensity: # FLOPs / # bytes

- If high, operation is compute-bound (good)

- If low, operation is memory-bound (bad)

General rule: matrix multiplication is compute-bound, everything else is memory-bound

IMPORTANT: benchmark/profile your code!

You can read spec sheets (marketing material) and papers

...but performance depends on your library version, your hardware, your workload

...so there is no substitute for benchmarking/profiling your code.

Example computation: running forward/backward passes on an MLP.

Benchmarking measures the wall-clock time of performing some operation.

It only gives you end-to-end time, not where time is spent (profiling).

It is still useful for:

- comparing different implementations (which is faster?), and

- understanding how performance scales (e.g., with dimension).

Let's define a convenient function for benchmarking an arbitrary function.

### Benchmarking matrix multiplication

First, let us benchmark matrix multiplication of square matrices.

Let us benchmark our MLP!

Scale the number of steps.

Scale the number of layers.

Scale the batch size.

Scale the dimension.

The timings are not always predictable due to the non-homogenous nature of CUDA kernels, hardware, etc.

You can also use `torch.utils.benchmark`, which provides more amenities.

[https://pytorch.org/tutorials/recipes/recipes/benchmark.html](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)

We did not use this to make benchmarking more transparent.

While benchmarking looks at end-to-end time, profiling looks at where time is spent.

Obvious: profiling helps you understand where time is being spent.

Deeper: profiling helps you understand (what is being called).

PyTorch has a nice built-in profiler [https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

Let's profile some code to see what is going on under the hood.

## sleep

-------------------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls -------------------------  ------------  ------------  ------------  ------------  ------------  ------------ cudaDeviceSynchronize       100.00%      11.610us       100.00%      11.610us       5.805us             2 -------------------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 11.610us

Let's start with some basic operations.

## add

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::add        98.02%       1.392ms        99.38%       1.411ms       1.411ms      17.119us       100.00%      17.119us      17.119us             1 void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add...         0.00%       0.000us         0.00%       0.000us       0.000us      17.119us       100.00%      17.119us      17.119us             1 cudaLaunchKernel         1.37%      19.392us         1.37%      19.392us      19.392us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize         0.62%       8.734us         0.62%       8.734us       4.367us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.420ms Self CUDA time total: 17.119us

## matmul

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::matmul         2.29%       7.520us        97.24%     318.634us     318.634us       0.000us         0.00%     342.620us     342.620us             1 aten::mm        90.14%     295.387us        94.94%     311.114us     311.114us     342.620us       100.00%     342.620us     342.620us             1 void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(cutlass_80...         0.00%       0.000us         0.00%       0.000us       0.000us     342.620us       100.00%     342.620us     342.620us             1 cudaDeviceGetAttribute         0.21%       0.690us         0.21%       0.690us       0.690us       0.000us         0.00%       0.000us       0.000us             1 cuLaunchKernel         4.59%      15.037us         4.59%      15.037us      15.037us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize         2.76%       9.051us         2.76%       9.051us       4.525us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 327.685us Self CUDA time total: 342.620us

## matmul(dim=128)

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::matmul         1.17%       4.912us        98.24%     413.723us     413.723us       0.000us         0.00%       4.992us       4.992us             1 aten::mm        42.40%     178.581us        97.07%     408.811us     408.811us       4.992us       100.00%       4.992us       4.992us             1 sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_stage3_warpsize1x2x1_ff...         0.00%       0.000us         0.00%       0.000us       0.000us       4.992us       100.00%       4.992us       4.992us             1 cudaFuncGetAttributes         0.96%       4.023us         0.96%       4.023us       4.023us       0.000us         0.00%       0.000us       0.000us             1 cudaLaunchKernelExC        53.71%     226.207us        53.71%     226.207us     226.207us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize         1.76%       7.413us         1.76%       7.413us       3.707us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 421.136us Self CUDA time total: 4.992us

Observations

- You can see what CUDA kernels are actually being called.

- Different CUDA kernels are invoked depending on the tensor dimensions.

Name of CUDA kernel tells us something about the implementation.

Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1

- cutlass: NVIDIA's CUDA library for linear algebra

- 256x128: tile size

Let's now look at some composite operations.

## cdist

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::cdist         1.38%      27.430us        99.62%       1.982ms       1.982ms       0.000us         0.00%     440.121us     440.121us             1 aten::_euclidean_dist         2.92%      58.128us        97.28%       1.936ms       1.936ms       0.000us         0.00%     440.121us     440.121us             1 aten::matmul         0.10%       1.961us         2.51%      49.898us      49.898us       0.000us         0.00%     343.740us     343.740us             1 aten::mm         1.92%      38.220us         2.41%      47.937us      47.937us     343.740us        78.10%     343.740us     343.740us             1 sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_...         0.00%       0.000us         0.00%       0.000us       0.000us     343.740us        78.10%     343.740us     343.740us             1 aten::cat         0.88%      17.459us         1.33%      26.451us      13.226us      29.888us         6.79%      29.888us      14.944us             2 void at::native::(anonymous namespace)::CatArrayBatchedCopy_aligned16_contig<...         0.00%       0.000us         0.00%       0.000us       0.000us      29.888us         6.79%      29.888us      14.944us             2 aten::pow        72.92%       1.451ms        84.00%       1.671ms     835.719us      22.304us         5.07%      22.304us      11.152us             2 void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous name...         0.00%       0.000us         0.00%       0.000us       0.000us      22.304us         5.07%      22.304us      11.152us             2 aten::sum         1.22%      24.211us         1.77%      35.272us      17.636us      14.976us         3.40%      14.976us       7.488us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.990ms Self CUDA time total: 440.121us

## gelu

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::add        86.27%       1.422ms        98.31%       1.620ms       1.620ms      18.304us        66.90%      18.304us      18.304us             1 void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add...         0.00%       0.000us         0.00%       0.000us       0.000us      18.304us        66.90%      18.304us      18.304us             1 aten::gelu         0.74%      12.250us         1.13%      18.672us      18.672us       9.056us        33.10%       9.056us       9.056us             1 void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelI...         0.00%       0.000us         0.00%       0.000us       0.000us       9.056us        33.10%       9.056us       9.056us             1 cudaLaunchKernel        12.42%     204.811us        12.42%     204.811us     102.406us       0.000us         0.00%       0.000us       0.000us             2 cudaDeviceSynchronize         0.56%       9.236us         0.56%       9.236us       4.618us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.648ms Self CUDA time total: 27.360us

## softmax

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::softmax         0.70%      11.487us         1.85%      30.256us      30.256us       0.000us         0.00%      20.191us      20.191us             1 aten::_softmax         0.73%      11.951us         1.15%      18.769us      18.769us      20.191us        52.15%      20.191us      20.191us             1 void at::native::(anonymous namespace)::cunn_SoftMaxForwardSmem<4, float, flo...         0.00%       0.000us         0.00%       0.000us       0.000us      20.191us        52.15%      20.191us      20.191us             1 aten::add        87.82%       1.434ms        97.71%       1.596ms       1.596ms      18.528us        47.85%      18.528us      18.528us             1 void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add...         0.00%       0.000us         0.00%       0.000us       0.000us      18.528us        47.85%      18.528us      18.528us             1 cudaLaunchKernel        10.31%     168.454us        10.31%     168.454us      84.227us       0.000us         0.00%       0.000us       0.000us             2 cudaDeviceSynchronize         0.43%       7.097us         0.43%       7.097us       3.549us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.633ms Self CUDA time total: 38.719us

Now let's profile our MLP.

We will also visualize our stack trace using a flame graph, which reveals where time is being spent.

## mlp

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ autograd::engine::evaluate_function: AddmmBackward0         1.72%     672.089us        16.79%       6.571ms      51.339us       0.000us         0.00%      46.088ms     360.062us           128 AddmmBackward0         1.32%     517.317us        10.87%       4.253ms      33.224us       0.000us         0.00%      44.625ms     348.634us           128 aten::mm         5.47%       2.141ms         7.96%       3.115ms      12.263us      44.625ms        60.63%      44.625ms     175.690us           254 aten::linear         0.55%     217.124us        12.51%       4.895ms      38.245us       0.000us         0.00%      23.450ms     183.203us           128 aten::addmm         7.09%       2.773ms        10.67%       4.176ms      32.621us      23.450ms        31.86%      23.450ms     183.203us           128 sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_...         0.00%       0.000us         0.00%       0.000us       0.000us      23.450ms        31.86%      23.450ms     183.203us           128 void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(cutlass_80...         0.00%       0.000us         0.00%       0.000us       0.000us      21.923ms        29.79%      21.923ms     173.991us           126 void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(cutlass_80...         0.00%       0.000us         0.00%       0.000us       0.000us      21.741ms        29.54%      21.741ms     169.850us           128 autograd::engine::evaluate_function: torch::autograd::AccumulateGrad         1.34%     522.511us         5.72%       2.238ms       8.743us       0.000us         0.00%       1.962ms       7.663us           256 torch::autograd::AccumulateGrad         0.88%     342.816us         4.38%       1.716ms       6.702us       0.000us         0.00%       1.962ms       7.663us           256 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 39.129ms Self CUDA time total: 73.598ms

Every time you make a change, benchmark/profile!

Horace He's blog post [[Article]](https://horace.io/brrr_intro.html)

Analogy: warehouse : DRAM :: factory : SRAM

<img src="../../materials/spring2025-lectures/var/files/image-27f5a4326a831bfe8b3af774827dd675-https_horace_io_img_perf_intro_factory_bandwidth_png" width="800">

Each operation needs to read/compute/write:

<img src="../../materials/spring2025-lectures/var/files/image-cc6ef17556dcaa1009d789500b39990e-https_horace_io_img_perf_intro_multi_operators_png" width="800">

If we *fuse* the operations, only need to read/write once:

<img src="../../materials/spring2025-lectures/var/files/image-31427825beca763b76d3700e85b1065e-https_horace_io_img_perf_intro_operator_fusion_png" width="800">

To see the effect of fusion, let's consider the GeLU activation function.

[https://pytorch.org/docs/stable/generated/torch.nn.GELU.html](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)

Let's consider two ways to compute GeLU:

1. The default PyTorch implementation (fused):

2. We can also write our own by hand (not fused):

Let's benchmark.

The fused version is significantly faster: 8.15 ms, 1.13 ms

Let's look under the hood.

## manual_gelu

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::mul        15.19%       1.479ms        26.01%       2.532ms     422.074us       5.222ms        68.10%       5.222ms     870.388us             6 void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<f...         0.00%       0.000us         0.00%       0.000us       0.000us       3.112ms        40.58%       3.112ms       1.037ms             3 void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<f...         0.00%       0.000us         0.00%       0.000us       0.000us       2.110ms        27.51%       2.110ms     703.350us             3 aten::add         0.13%      12.473us         0.21%      20.089us      10.045us       1.742ms        22.72%       1.742ms     871.157us             2 void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add...         0.00%       0.000us         0.00%       0.000us       0.000us       1.036ms        13.51%       1.036ms       1.036ms             1 void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSe...         0.00%       0.000us         0.00%       0.000us       0.000us     706.423us         9.21%     706.423us     706.423us             1 aten::tanh         0.07%       6.961us         0.12%      11.588us      11.588us     704.343us         9.18%     704.343us     704.343us             1 void at::native::vectorized_elementwise_kernel<4, at::native::tanh_kernel_cud...         0.00%       0.000us         0.00%       0.000us       0.000us     704.343us         9.18%     704.343us     704.343us             1 cudaLaunchKernel        10.95%       1.066ms        10.95%       1.066ms     118.432us       0.000us         0.00%       0.000us       0.000us             9 cudaDeviceSynchronize        73.66%       7.171ms        73.66%       7.171ms       3.586ms       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 9.735ms Self CUDA time total: 7.669ms

## pytorch_gelu

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::gelu        71.12%       1.436ms        84.28%       1.701ms       1.701ms     701.560us       100.00%     701.560us     701.560us             1 void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelI...         0.00%       0.000us         0.00%       0.000us       0.000us     701.560us       100.00%     701.560us     701.560us             1 cudaLaunchKernel        13.16%     265.687us        13.16%     265.687us     265.687us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize        15.72%     317.405us        15.72%     317.405us     158.703us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 2.019ms Self CUDA time total: 701.560us

The PyTorch just calls one kernel whereas the others are atomic (remember the warehouse/factory)

## Look at Nsight profiler for MLP

Now let's open the box to understand what's going on inside a CUDA kernel by writing our own.

Let's write the GeLU function in CUDA.

CUDA is an extension of C/C++ with APIs for managing GPUs.

Simplified picture: write f(i), CUDA kernel computes f(i) for all i.

<img src="../../materials/spring2025-lectures/var/files/image-1390e250b108c0d04b315ac7995eccd0-https_docs_nvidia_com_cuda_parallel-thread-execution__images_grid-with-CTAs_png" width="0.5">

Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)

Thread block: collection of threads: blockIdx = (0, 1)

Thread: single unit of operation: threadIdx = (0, 3).

You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do.

Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.

The `load_inline` function makes it convenient to write CUDA code and bind it to a Python module for immediate use.

#include <math.h>

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) { // Get the index into the tensor int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < num_elements) {  // To handle the case when n < numBlocks * blockDim // Do the actual computation out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i]))); } } inline unsigned int cdiv(unsigned int a, unsigned int b) { // Compute ceil(a / b) return (a + b - 1) / b; } torch::Tensor gelu(torch::Tensor x) { TORCH_CHECK(x.device().is_cuda()); TORCH_CHECK(x.is_contiguous()); // Allocate empty tensor torch::Tensor y = torch::empty_like(x); // Determine grid (elements divided into blocks) int num_elements = x.numel(); int block_size = 1024;  // Number of threads int num_blocks = cdiv(num_elements, block_size); // Launch the kernel gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements); C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately return y; }

Compile the CUDA code and bind it to a Python module.

Check correctness of our implementation.

Benchmark our CUDA version.

## cuda_gelu

------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls ------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ gelu_kernel(float*, float*, int)         0.00%       0.000us         0.00%       0.000us       0.000us       1.664ms       100.00%       1.664ms       1.664ms             1 aten::empty_like         0.20%       6.209us        46.52%       1.428ms       1.428ms       0.000us         0.00%       0.000us       0.000us             1 aten::empty_strided        46.31%       1.422ms        46.31%       1.422ms       1.422ms       0.000us         0.00%       0.000us       0.000us             1 cudaLaunchKernel         8.93%     274.078us         8.93%     274.078us     274.078us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize        44.56%       1.368ms        44.56%       1.368ms     683.944us       0.000us         0.00%       0.000us       0.000us             2 ------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 3.070ms Self CUDA time total: 1.664ms

Our CUDA implementation is faster than manual, but not as good as PyTorch.

Elementwise operations are easy in CUDA (though you can still be smarter).

But most interesting operations (e.g., matmul, softmax, RMSNorm) require reading multiple values.

For that, you have to think about managing shared memory, etc.

Developed by OpenAI in 2021

[https://openai.com/research/triton](https://openai.com/research/triton)

Make GPU programming more accessible

- Write in Python

- Think about thread blocks rather than threads

What does Triton offer?

CUDA      Triton

- Memory coalescing (transfer from DRAM)     manual    automatic

- Shared memory management                   manual    automatic

- Scheduling within SMs                      manual    automatic

- Scheduling across SMs                      manual    manual

Compiler does more work, can actually outperform PyTorch implementations!

One big advantage of Triton is that you can step through the Python code.

Let's step through a Triton kernel.

PTX (parallel thread execution) is like an assembly language for GPUs.

We can see the PTX code generated by Triton.

[https://docs.nvidia.com/cuda/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

Let's go poke around at the PTX code.

[https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/triton_gelu-ptx.txt](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/triton_gelu-ptx.txt)

// // Generated by LLVM NVPTX Back-End // .version 8.4 .target sm_90a .address_size 64 // .globl	triton_gelu_kernel      // -- Begin function triton_gelu_kernel // @triton_gelu_kernel .visible .entry triton_gelu_kernel( .param .u64 .ptr .global .align 1 triton_gelu_kernel_param_0, .param .u64 .ptr .global .align 1 triton_gelu_kernel_param_1, .param .u32 triton_gelu_kernel_param_2 ) .reqntid 128, 1, 1 { .reg .pred 	%p<5>; .reg .b32 	%r<49>; .reg .f32 	%f<113>; .reg .b64 	%rd<8>; .loc	1 552 0                         // lecture_06.py:552:0 $L__func_begin0: .loc	1 552 0                         // lecture_06.py:552:0 // %bb.0: ld.param.u64 	%rd5, [triton_gelu_kernel_param_0]; ld.param.u64 	%rd6, [triton_gelu_kernel_param_1]; $L__tmp0: .loc	1 557 24                        // lecture_06.py:557:24 // begin inline asm mov.u32 %r1, %ctaid.x; // end inline asm .loc	1 558 24                        // lecture_06.py:558:24 shl.b32 	%r42, %r1, 10; ld.param.u32 	%r43, [triton_gelu_kernel_param_2]; .loc	1 561 41                        // lecture_06.py:561:41 mov.u32 	%r44, %tid.x; shl.b32 	%r45, %r44, 2; and.b32  	%r46, %r45, 508; .loc	1 561 28                        // lecture_06.py:561:28 or.b32  	%r47, %r42, %r46; or.b32  	%r48, %r47, 512; .loc	1 564 21                        // lecture_06.py:564:21 setp.lt.s32 	%p1, %r47, %r43; setp.lt.s32 	%p2, %r48, %r43; .loc	1 567 24                        // lecture_06.py:567:24 mul.wide.s32 	%rd7, %r47, 4; add.s64 	%rd1, %rd5, %rd7; add.s64 	%rd2, %rd1, 2048; .loc	1 567 16                        // lecture_06.py:567:16 // begin inline asm mov.u32 %r2, 0x0; mov.u32 %r3, 0x0; mov.u32 %r4, 0x0; mov.u32 %r5, 0x0; @%p1 ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd1 + 0 ]; // end inline asm mov.b32 	%f17, %r2; mov.b32 	%f18, %r3; mov.b32 	%f19, %r4; mov.b32 	%f20, %r5; // begin inline asm mov.u32 %r6, 0x0; mov.u32 %r7, 0x0; mov.u32 %r8, 0x0; mov.u32 %r9, 0x0; @%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ]; // end inline asm mov.b32 	%f21, %r6; mov.b32 	%f22, %r7; mov.b32 	%f23, %r8; mov.b32 	%f24, %r9; .loc	1 571 37                        // lecture_06.py:571:37 mul.f32 	%f25, %f17, 0f3D372713; mul.f32 	%f26, %f18, 0f3D372713; mul.f32 	%f27, %f19, 0f3D372713; mul.f32 	%f28, %f20, 0f3D372713; mul.f32 	%f29, %f21, 0f3D372713; mul.f32 	%f30, %f22, 0f3D372713; mul.f32 	%f31, %f23, 0f3D372713; mul.f32 	%f32, %f24, 0f3D372713; .loc	1 571 41                        // lecture_06.py:571:41 mul.f32 	%f33, %f25, %f17; mul.f32 	%f34, %f26, %f18; mul.f32 	%f35, %f27, %f19; mul.f32 	%f36, %f28, %f20; mul.f32 	%f37, %f29, %f21; mul.f32 	%f38, %f30, %f22; mul.f32 	%f39, %f31, %f23; mul.f32 	%f40, %f32, %f24; .loc	1 571 26                        // lecture_06.py:571:26 fma.rn.f32 	%f41, %f33, %f17, %f17; fma.rn.f32 	%f42, %f34, %f18, %f18; fma.rn.f32 	%f43, %f35, %f19, %f19; fma.rn.f32 	%f44, %f36, %f20, %f20; fma.rn.f32 	%f45, %f37, %f21, %f21; fma.rn.f32 	%f46, %f38, %f22, %f22; fma.rn.f32 	%f47, %f39, %f23, %f23; fma.rn.f32 	%f48, %f40, %f24, %f24; .loc	1 571 22                        // lecture_06.py:571:22 mul.f32 	%f49, %f41, 0f3F4C422A; mul.f32 	%f50, %f42, 0f3F4C422A; mul.f32 	%f51, %f43, 0f3F4C422A; mul.f32 	%f52, %f44, 0f3F4C422A; mul.f32 	%f53, %f45, 0f3F4C422A; mul.f32 	%f54, %f46, 0f3F4C422A; mul.f32 	%f55, %f47, 0f3F4C422A; mul.f32 	%f56, %f48, 0f3F4C422A; .loc	1 572 21                        // lecture_06.py:572:21 fma.rn.f32 	%f57, %f41, 0f3F4C422A, %f49; fma.rn.f32 	%f58, %f42, 0f3F4C422A, %f50; fma.rn.f32 	%f59, %f43, 0f3F4C422A, %f51; fma.rn.f32 	%f60, %f44, 0f3F4C422A, %f52; fma.rn.f32 	%f61, %f45, 0f3F4C422A, %f53; fma.rn.f32 	%f62, %f46, 0f3F4C422A, %f54; fma.rn.f32 	%f63, %f47, 0f3F4C422A, %f55; fma.rn.f32 	%f64, %f48, 0f3F4C422A, %f56; .loc	1 572 17                        // lecture_06.py:572:17 mul.f32 	%f2, %f57, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f1, %f2; // end inline asm mul.f32 	%f4, %f58, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f3, %f4; // end inline asm mul.f32 	%f6, %f59, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f5, %f6; // end inline asm mul.f32 	%f8, %f60, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f7, %f8; // end inline asm mul.f32 	%f10, %f61, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f9, %f10; // end inline asm mul.f32 	%f12, %f62, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f11, %f12; // end inline asm mul.f32 	%f14, %f63, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f13, %f14; // end inline asm mul.f32 	%f16, %f64, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f15, %f16; // end inline asm .loc	1 573 18                        // lecture_06.py:573:18 add.f32 	%f65, %f1, 0fBF800000; add.f32 	%f66, %f3, 0fBF800000; add.f32 	%f67, %f5, 0fBF800000; add.f32 	%f68, %f7, 0fBF800000; add.f32 	%f69, %f9, 0fBF800000; add.f32 	%f70, %f11, 0fBF800000; add.f32 	%f71, %f13, 0fBF800000; add.f32 	%f72, %f15, 0fBF800000; .loc	1 573 30                        // lecture_06.py:573:30 add.f32 	%f73, %f1, 0f3F800000; add.f32 	%f74, %f3, 0f3F800000; add.f32 	%f75, %f5, 0f3F800000; add.f32 	%f76, %f7, 0f3F800000; add.f32 	%f77, %f9, 0f3F800000; add.f32 	%f78, %f11, 0f3F800000; add.f32 	%f79, %f13, 0f3F800000; add.f32 	%f80, %f15, 0f3F800000; .loc	1 573 24                        // lecture_06.py:573:24 mov.b32 	%r11, %f65; mov.b32 	%r12, %f73; // begin inline asm div.full.f32 %r10, %r11, %r12; // end inline asm mov.b32 	%f81, %r10; mov.b32 	%r14, %f66; mov.b32 	%r15, %f74; // begin inline asm div.full.f32 %r13, %r14, %r15; // end inline asm mov.b32 	%f82, %r13; mov.b32 	%r17, %f67; mov.b32 	%r18, %f75; // begin inline asm div.full.f32 %r16, %r17, %r18; // end inline asm mov.b32 	%f83, %r16; mov.b32 	%r20, %f68; mov.b32 	%r21, %f76; // begin inline asm div.full.f32 %r19, %r20, %r21; // end inline asm mov.b32 	%f84, %r19; mov.b32 	%r23, %f69; mov.b32 	%r24, %f77; // begin inline asm div.full.f32 %r22, %r23, %r24; // end inline asm mov.b32 	%f85, %r22; mov.b32 	%r26, %f70; mov.b32 	%r27, %f78; // begin inline asm div.full.f32 %r25, %r26, %r27; // end inline asm mov.b32 	%f86, %r25; mov.b32 	%r29, %f71; mov.b32 	%r30, %f79; // begin inline asm div.full.f32 %r28, %r29, %r30; // end inline asm mov.b32 	%f87, %r28; mov.b32 	%r32, %f72; mov.b32 	%r33, %f80; // begin inline asm div.full.f32 %r31, %r32, %r33; // end inline asm mov.b32 	%f88, %r31; .loc	1 574 14                        // lecture_06.py:574:14 mul.f32 	%f89, %f17, 0f3F000000; mul.f32 	%f90, %f18, 0f3F000000; mul.f32 	%f91, %f19, 0f3F000000; mul.f32 	%f92, %f20, 0f3F000000; mul.f32 	%f93, %f21, 0f3F000000; mul.f32 	%f94, %f22, 0f3F000000; mul.f32 	%f95, %f23, 0f3F000000; mul.f32 	%f96, %f24, 0f3F000000; .loc	1 574 23                        // lecture_06.py:574:23 add.f32 	%f97, %f81, 0f3F800000; add.f32 	%f98, %f82, 0f3F800000; add.f32 	%f99, %f83, 0f3F800000; add.f32 	%f100, %f84, 0f3F800000; add.f32 	%f101, %f85, 0f3F800000; add.f32 	%f102, %f86, 0f3F800000; add.f32 	%f103, %f87, 0f3F800000; add.f32 	%f104, %f88, 0f3F800000; .loc	1 574 19                        // lecture_06.py:574:19 mul.f32 	%f105, %f89, %f97; mul.f32 	%f106, %f90, %f98; mul.f32 	%f107, %f91, %f99; mul.f32 	%f108, %f92, %f100; mul.f32 	%f109, %f93, %f101; mul.f32 	%f110, %f94, %f102; mul.f32 	%f111, %f95, %f103; mul.f32 	%f112, %f96, %f104; .loc	1 577 21                        // lecture_06.py:577:21 add.s64 	%rd3, %rd6, %rd7; add.s64 	%rd4, %rd3, 2048; .loc	1 577 30                        // lecture_06.py:577:30 mov.b32 	%r34, %f105; mov.b32 	%r35, %f106; mov.b32 	%r36, %f107; mov.b32 	%r37, %f108; // begin inline asm @%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r34, %r35, %r36, %r37 }; // end inline asm mov.b32 	%r38, %f109; mov.b32 	%r39, %f110; mov.b32 	%r40, %f111; mov.b32 	%r41, %f112; // begin inline asm @%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r38, %r39, %r40, %r41 }; // end inline asm .loc	1 577 4                         // lecture_06.py:577:4 ret; $L__tmp1: $L__func_end0: // -- End function } .file	1 "/home/c-thashim/2025/spring2025-lectures/lecture_06.py" .section	.debug_abbrev { .b8 1                                   // Abbreviation Code .b8 17                                  // DW_TAG_compile_unit .b8 0                                   // DW_CHILDREN_no .b8 37                                  // DW_AT_producer .b8 8                                   // DW_FORM_string .b8 19                                  // DW_AT_language .b8 5                                   // DW_FORM_data2 .b8 3                                   // DW_AT_name .b8 8                                   // DW_FORM_string .b8 16                                  // DW_AT_stmt_list .b8 6                                   // DW_FORM_data4 .b8 27                                  // DW_AT_comp_dir .b8 8                                   // DW_FORM_string .b8 0                                   // EOM(1) .b8 0                                   // EOM(2) .b8 0                                   // EOM(3) } .section	.debug_info { .b32 76                                 // Length of Unit .b8 2                                   // DWARF version number .b8 0 .b32 .debug_abbrev                      // Offset Into Abbrev. Section .b8 8                                   // Address Size (in bytes) .b8 1                                   // Abbrev [1] 0xb:0x45 DW_TAG_compile_unit .b8 116                                 // DW_AT_producer .b8 114 .b8 105 .b8 116 .b8 111 .b8 110 .b8 0 .b8 2                                   // DW_AT_language .b8 0 .b8 108                                 // DW_AT_name .b8 101 .b8 99 .b8 116 .b8 117 .b8 114 .b8 101 .b8 95 .b8 48 .b8 54 .b8 46 .b8 112 .b8 121 .b8 0 .b32 .debug_line                        // DW_AT_stmt_list .b8 47                                  // DW_AT_comp_dir .b8 104 .b8 111 .b8 109 .b8 101 .b8 47 .b8 99 .b8 45 .b8 116 .b8 104 .b8 97 .b8 115 .b8 104 .b8 105 .b8 109 .b8 47 .b8 50 .b8 48 .b8 50 .b8 53 .b8 47 .b8 115 .b8 112 .b8 114 .b8 105 .b8 110 .b8 103 .b8 50 .b8 48 .b8 50 .b8 53 .b8 45 .b8 108 .b8 101 .b8 99 .b8 116 .b8 117 .b8 114 .b8 101 .b8 115 .b8 0 } .section	.debug_macinfo	{	}

Observations:

- ld.global.* and st.global.* reads and writes from global memory

- %ctaid.x is block index, %tid.x is thread index

- %f* are floating point registers, %r* are integer registers

- One thread processes 8 elements at the same time (thread coarsening)

Check that it's correct.

Let's now benchmark it compared to the PyTorch and CUDA implementations.

Remember to set TRITON_INTERPRET=0 for good performance.

CUDA is an extension of C/C++ with APIs for managing GPUs.

Simplified picture: write f(i), CUDA kernel computes f(i) for all i.

<img src="../../materials/spring2025-lectures/var/files/image-1390e250b108c0d04b315ac7995eccd0-https_docs_nvidia_com_cuda_parallel-thread-execution__images_grid-with-CTAs_png" width="0.5">

Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)

Thread block: collection of threads: blockIdx = (0, 1)

Thread: single unit of operation: threadIdx = (0, 3).

You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do.

Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.

The `load_inline` function makes it convenient to write CUDA code and bind it to a Python module for immediate use.

#include <math.h>

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) { // Get the index into the tensor int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < num_elements) {  // To handle the case when n < numBlocks * blockDim // Do the actual computation out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i]))); } } inline unsigned int cdiv(unsigned int a, unsigned int b) { // Compute ceil(a / b) return (a + b - 1) / b; } torch::Tensor gelu(torch::Tensor x) { TORCH_CHECK(x.device().is_cuda()); TORCH_CHECK(x.is_contiguous()); // Allocate empty tensor torch::Tensor y = torch::empty_like(x); // Determine grid (elements divided into blocks) int num_elements = x.numel(); int block_size = 1024;  // Number of threads int num_blocks = cdiv(num_elements, block_size); // Launch the kernel gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements); C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately return y; }

Compile the CUDA code and bind it to a Python module.

## triton_gelu

-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ triton_gelu_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     705.240us       100.00%     705.240us     705.240us             1 aten::empty_like         0.32%       5.629us        72.49%       1.267ms       1.267ms       0.000us         0.00%       0.000us       0.000us             1 aten::empty_strided        72.16%       1.261ms        72.16%       1.261ms       1.261ms       0.000us         0.00%       0.000us       0.000us             1 cuLaunchKernel        16.46%     287.632us        16.46%     287.632us     287.632us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize        11.05%     193.144us        11.05%     193.144us      96.572us       0.000us         0.00%       0.000us       0.000us             2 -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.747ms Self CUDA time total: 705.240us

Our Triton implementation (triton_gelu):

- is almost as good as the PyTorch implementation (pytorch_gelu).

- is actually slower than our naive CUDA implementation (cuda_gelu).

Triton operates on blocks, CUDA operates on threads.

Blocks allows Triton compiler to do other optimizations (e.g., thread coarsening).

Everything is way faster than the manual implementation (manual_gelu).

So far, we have seen three ways to write GeLU:

- Use the default PyTorch function

- Write it in Python `manual_gelu` (lecture_06.py:868)

- Write it in CUDA `create_cuda_gelu` (lecture_06.py:433)

- Write it in Triton `triton_gelu` (lecture_06.py:534)

- Write it in Python and compile it into Triton

Check correctness of our implementation.

Let's benchmark and profile it!

CUDA is an extension of C/C++ with APIs for managing GPUs.

Simplified picture: write f(i), CUDA kernel computes f(i) for all i.

<img src="../../materials/spring2025-lectures/var/files/image-1390e250b108c0d04b315ac7995eccd0-https_docs_nvidia_com_cuda_parallel-thread-execution__images_grid-with-CTAs_png" width="0.5">

Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)

Thread block: collection of threads: blockIdx = (0, 1)

Thread: single unit of operation: threadIdx = (0, 3).

You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do.

Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.

The `load_inline` function makes it convenient to write CUDA code and bind it to a Python module for immediate use.

#include <math.h>

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) { // Get the index into the tensor int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < num_elements) {  // To handle the case when n < numBlocks * blockDim // Do the actual computation out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i]))); } } inline unsigned int cdiv(unsigned int a, unsigned int b) { // Compute ceil(a / b) return (a + b - 1) / b; } torch::Tensor gelu(torch::Tensor x) { TORCH_CHECK(x.device().is_cuda()); TORCH_CHECK(x.is_contiguous()); // Allocate empty tensor torch::Tensor y = torch::empty_like(x); // Determine grid (elements divided into blocks) int num_elements = x.numel(); int block_size = 1024;  // Number of threads int num_blocks = cdiv(num_elements, block_size); // Launch the kernel gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements); C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately return y; }

Compile the CUDA code and bind it to a Python module.

Let's look under the hood

## compiled_gelu

-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls -----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Torch-Compiled Region: 0/1        77.92%       1.934ms        91.35%       2.268ms       2.268ms       0.000us         0.00%     707.261us     707.261us             1 triton_poi_fused_add_mul_tanh_0         1.84%      45.795us        13.43%     333.455us     333.455us     707.261us       100.00%     707.261us     707.261us             1 triton_poi_fused_add_mul_tanh_0         0.00%       0.000us         0.00%       0.000us       0.000us     707.261us       100.00%     707.261us     707.261us             1 TorchDynamo Cache Lookup         0.62%      15.371us         0.62%      15.371us      15.371us       0.000us         0.00%       0.000us       0.000us             1 cuLaunchKernel        11.59%     287.660us        11.59%     287.660us     287.660us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize         8.03%     199.299us         8.03%     199.299us      99.649us       0.000us         0.00%       0.000us       0.000us             2 -----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 2.482ms Self CUDA time total: 707.261us

So far, we've looked at elementwise operations in Triton (e.g., GeLU).

Now let us look at operations that aggregate over multiple values.

We will roughly follow the Triton fused softmax tutorial: [https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

Recall the softmax operation is used in attention and generating probabilities.

Normalize each row of a matrix:

[A1 A2 A3]   =>   [A1/A A2/A A3/A]

[B1 B2 B3]   =>   [B1/B B2/B B3/B]

Let's first start with the naive implementation and keep track of reads/writes.

Now let us write the Triton kernel.

Check our implementations are correct.

Now let's benchmark everything.

Look under the hood using the profiler.

## manual_softmax

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::div         0.28%       8.466us         0.43%      13.241us      13.241us     921.170us        28.28%     921.170us     921.170us             1 void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocas...         0.00%       0.000us         0.00%       0.000us       0.000us     921.170us        28.28%     921.170us     921.170us             1 aten::sub         0.44%      13.364us         0.70%      21.601us      21.601us     892.243us        27.39%     892.243us     892.243us             1 void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocas...         0.00%       0.000us         0.00%       0.000us       0.000us     892.243us        27.39%     892.243us     892.243us             1 aten::exp         0.25%       7.610us         0.41%      12.595us      12.595us     703.925us        21.61%     703.925us     703.925us             1 void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda...         0.00%       0.000us         0.00%       0.000us       0.000us     703.925us        21.61%     703.925us     703.925us             1 aten::max        10.71%     328.962us        19.95%     612.839us     612.839us     393.433us        12.08%     393.433us     393.433us             1 void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native...         0.00%       0.000us         0.00%       0.000us       0.000us     393.433us        12.08%     393.433us     393.433us             1 aten::sum         0.38%      11.798us         0.67%      20.683us      20.683us     346.779us        10.65%     346.779us     346.779us             1 void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native...         0.00%       0.000us         0.00%       0.000us       0.000us     346.779us        10.65%     346.779us     346.779us             1 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 3.071ms Self CUDA time total: 3.258ms

## compiled_softmax

------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls ------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Torch-Compiled Region: 1/0        56.77%     769.985us        78.99%       1.071ms       1.071ms       0.000us         0.00%     730.770us     730.770us             1 triton_red_fused_div_exp_max_sub_sum_0         3.20%      43.382us        22.22%     301.366us     301.366us     730.770us       100.00%     730.770us     730.770us             1 triton_red_fused_div_exp_max_sub_sum_0         0.00%       0.000us         0.00%       0.000us       0.000us     730.770us       100.00%     730.770us     730.770us             1 TorchDynamo Cache Lookup         0.53%       7.239us         0.53%       7.239us       7.239us       0.000us         0.00%       0.000us       0.000us             1 cuLaunchKernel        19.02%     257.984us        19.02%     257.984us     257.984us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize        20.48%     277.800us        20.48%     277.800us     138.900us       0.000us         0.00%       0.000us       0.000us             2 ------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.356ms Self CUDA time total: 730.770us

## pytorch_softmax

--------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ aten::softmax         0.47%       5.061us        28.87%     312.564us     312.564us       0.000us         0.00%       1.137ms       1.137ms             1 aten::_softmax        13.44%     145.534us        28.40%     307.503us     307.503us       1.137ms       100.00%       1.137ms       1.137ms             1 void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.137ms       100.00%       1.137ms       1.137ms             1 cudaLaunchKernel        14.96%     161.969us        14.96%     161.969us     161.969us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize        71.13%     770.228us        71.13%     770.228us     385.114us       0.000us         0.00%       0.000us       0.000us             2 --------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.083ms Self CUDA time total: 1.137ms

## triton_softmax

-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ triton_softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     705.462us       100.00%     705.462us     705.462us             1 aten::empty_like         0.23%       4.238us        77.57%       1.409ms       1.409ms       0.000us         0.00%       0.000us       0.000us             1 aten::empty_strided        77.34%       1.405ms        77.34%       1.405ms       1.405ms       0.000us         0.00%       0.000us       0.000us             1 cuLaunchKernel         8.02%     145.738us         8.02%     145.738us     145.738us       0.000us         0.00%       0.000us       0.000us             1 cudaDeviceSynchronize        14.41%     261.640us        14.41%     261.640us     130.820us       0.000us         0.00%       0.000us       0.000us             2 -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ Self CPU time total: 1.816ms Self CUDA time total: 705.462us

Let's end by looking at the PTX code.

Let's go poke around at the PTX code.

[https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/triton_softmax-ptx.txt](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/triton_softmax-ptx.txt)

// // Generated by LLVM NVPTX Back-End // .version 8.4 .target sm_90a .address_size 64 // .globl	triton_softmax_kernel   // -- Begin function triton_softmax_kernel // @triton_softmax_kernel .visible .entry triton_softmax_kernel( .param .u64 .ptr .global .align 1 triton_softmax_kernel_param_0, .param .u64 .ptr .global .align 1 triton_softmax_kernel_param_1, .param .u32 triton_softmax_kernel_param_2, .param .u32 triton_softmax_kernel_param_3, .param .u32 triton_softmax_kernel_param_4 ) .reqntid 128, 1, 1 { .reg .pred 	%p<5>; .reg .b32 	%r<22>; .reg .f32 	%f<13>; .reg .b64 	%rd<10>; .loc	1 741 0                         // lecture_06.py:741:0 $L__func_begin0: .loc	1 741 0                         // lecture_06.py:741:0 // %bb.0: ld.param.u64 	%rd3, [triton_softmax_kernel_param_0]; ld.param.u64 	%rd4, [triton_softmax_kernel_param_1]; $L__tmp0: .loc	1 745 28                        // lecture_06.py:745:28 // begin inline asm mov.u32 %r1, %ctaid.x; // end inline asm ld.param.u32 	%r8, [triton_softmax_kernel_param_2]; .loc	1 746 31                        // lecture_06.py:746:31 mov.u32 	%r9, %tid.x; and.b32  	%r10, %r9, 3; ld.param.u32 	%r11, [triton_softmax_kernel_param_3]; .loc	1 749 36                        // lecture_06.py:749:36 mul.lo.s32 	%r12, %r1, %r8; ld.param.u32 	%r13, [triton_softmax_kernel_param_4]; .loc	1 749 26                        // lecture_06.py:749:26 mul.wide.s32 	%rd5, %r12, 4; add.s64 	%rd6, %rd3, %rd5; .loc	1 750 27                        // lecture_06.py:750:27 mul.wide.u32 	%rd7, %r10, 4; add.s64 	%rd1, %rd6, %rd7; .loc	1 751 47                        // lecture_06.py:751:47 setp.lt.s32 	%p1, %r10, %r13; mov.b32 	%r3, -8388608; .loc	1 751 20                        // lecture_06.py:751:20 // begin inline asm mov.u32 %r2, 0x0; @%p1 ld.global.b32 { %r2 }, [ %rd1 + 0 ]; @!%p1 mov.u32 %r2, %r3; // end inline asm mov.b32 	%f3, %r2; $L__tmp1: .loc	2 184 40                        // standard.py:184:40 shfl.sync.bfly.b32	%r14, %r2, 2, 31, -1; mov.b32 	%f4, %r14; .loc	2 163 27                        // standard.py:163:27 max.f32 	%f5, %f3, %f4; .loc	2 184 40                        // standard.py:184:40 mov.b32 	%r15, %f5; shfl.sync.bfly.b32	%r16, %r15, 1, 31, -1; mov.b32 	%f6, %r16; .loc	2 163 27                        // standard.py:163:27 max.f32 	%f7, %f5, %f6; $L__tmp2: .loc	1 754 20                        // lecture_06.py:754:20 sub.f32 	%f8, %f3, %f7; .loc	1 755 23                        // lecture_06.py:755:23 mul.f32 	%f2, %f8, 0f3FB8AA3B; // begin inline asm ex2.approx.f32 %f1, %f2; // end inline asm $L__tmp3: .loc	2 267 36                        // standard.py:267:36 mov.b32 	%r5, %f1; shfl.sync.bfly.b32	%r17, %r5, 2, 31, -1; mov.b32 	%f9, %r17; .loc	2 256 15                        // standard.py:256:15 add.f32 	%f10, %f1, %f9; .loc	2 267 36                        // standard.py:267:36 mov.b32 	%r18, %f10; shfl.sync.bfly.b32	%r19, %r18, 1, 31, -1; mov.b32 	%f11, %r19; .loc	2 256 15                        // standard.py:256:15 add.f32 	%f12, %f10, %f11; $L__tmp4: .loc	1 757 24                        // lecture_06.py:757:24 mov.b32 	%r6, %f12; // begin inline asm div.full.f32 %r7, %r5, %r6; // end inline asm .loc	1 760 36                        // lecture_06.py:760:36 mul.lo.s32 	%r20, %r1, %r11; .loc	1 760 26                        // lecture_06.py:760:26 mul.wide.s32 	%rd8, %r20, 4; add.s64 	%rd9, %rd4, %rd8; .loc	1 761 27                        // lecture_06.py:761:27 add.s64 	%rd2, %rd9, %rd7; .loc	1 762 21                        // lecture_06.py:762:21 and.b32  	%r21, %r9, 124; setp.eq.s32 	%p4, %r21, 0; and.pred  	%p3, %p4, %p1; // begin inline asm @%p3 st.global.b32 [ %rd2 + 0 ], { %r7 }; // end inline asm .loc	1 762 4                         // lecture_06.py:762:4 ret; $L__tmp5: $L__func_end0: // -- End function } .file	1 "/home/c-thashim/2025/spring2025-lectures/lecture_06.py" .file	2 "/home/c-thashim/2025/spring2025-lectures/.venv/lib/python3.10/site-packages/triton/language/standard.py" .section	.debug_abbrev { .b8 1                                   // Abbreviation Code .b8 17                                  // DW_TAG_compile_unit .b8 1                                   // DW_CHILDREN_yes .b8 37                                  // DW_AT_producer .b8 8                                   // DW_FORM_string .b8 19                                  // DW_AT_language .b8 5                                   // DW_FORM_data2 .b8 3                                   // DW_AT_name .b8 8                                   // DW_FORM_string .b8 16                                  // DW_AT_stmt_list .b8 6                                   // DW_FORM_data4 .b8 27                                  // DW_AT_comp_dir .b8 8                                   // DW_FORM_string .b8 0                                   // EOM(1) .b8 0                                   // EOM(2) .b8 2                                   // Abbreviation Code .b8 46                                  // DW_TAG_subprogram .b8 0                                   // DW_CHILDREN_no .b8 3                                   // DW_AT_name .b8 8                                   // DW_FORM_string .b8 32                                  // DW_AT_inline .b8 11                                  // DW_FORM_data1 .b8 0                                   // EOM(1) .b8 0                                   // EOM(2) .b8 3                                   // Abbreviation Code .b8 46                                  // DW_TAG_subprogram .b8 1                                   // DW_CHILDREN_yes .b8 17                                  // DW_AT_low_pc .b8 1                                   // DW_FORM_addr .b8 18                                  // DW_AT_high_pc .b8 1                                   // DW_FORM_addr .b8 49                                  // DW_AT_abstract_origin .b8 19                                  // DW_FORM_ref4 .b8 0                                   // EOM(1) .b8 0                                   // EOM(2) .b8 4                                   // Abbreviation Code .b8 29                                  // DW_TAG_inlined_subroutine .b8 0                                   // DW_CHILDREN_no .b8 49                                  // DW_AT_abstract_origin .b8 19                                  // DW_FORM_ref4 .b8 17                                  // DW_AT_low_pc .b8 1                                   // DW_FORM_addr .b8 18                                  // DW_AT_high_pc .b8 1                                   // DW_FORM_addr .b8 88                                  // DW_AT_call_file .b8 11                                  // DW_FORM_data1 .b8 89                                  // DW_AT_call_line .b8 5                                   // DW_FORM_data2 .b8 87                                  // DW_AT_call_column .b8 11                                  // DW_FORM_data1 .b8 0                                   // EOM(1) .b8 0                                   // EOM(2) .b8 0                                   // EOM(3) } .section	.debug_info { .b32 173                                // Length of Unit .b8 2                                   // DWARF version number .b8 0 .b32 .debug_abbrev                      // Offset Into Abbrev. Section .b8 8                                   // Address Size (in bytes) .b8 1                                   // Abbrev [1] 0xb:0xa6 DW_TAG_compile_unit .b8 116                                 // DW_AT_producer .b8 114 .b8 105 .b8 116 .b8 111 .b8 110 .b8 0 .b8 2                                   // DW_AT_language .b8 0 .b8 108                                 // DW_AT_name .b8 101 .b8 99 .b8 116 .b8 117 .b8 114 .b8 101 .b8 95 .b8 48 .b8 54 .b8 46 .b8 112 .b8 121 .b8 0 .b32 .debug_line                        // DW_AT_stmt_list .b8 47                                  // DW_AT_comp_dir .b8 104 .b8 111 .b8 109 .b8 101 .b8 47 .b8 99 .b8 45 .b8 116 .b8 104 .b8 97 .b8 115 .b8 104 .b8 105 .b8 109 .b8 47 .b8 50 .b8 48 .b8 50 .b8 53 .b8 47 .b8 115 .b8 112 .b8 114 .b8 105 .b8 110 .b8 103 .b8 50 .b8 48 .b8 50 .b8 53 .b8 45 .b8 108 .b8 101 .b8 99 .b8 116 .b8 117 .b8 114 .b8 101 .b8 115 .b8 0 .b8 2                                   // Abbrev [2] 0x50:0x18 DW_TAG_subprogram .b8 116                                 // DW_AT_name .b8 114 .b8 105 .b8 116 .b8 111 .b8 110 .b8 95 .b8 115 .b8 111 .b8 102 .b8 116 .b8 109 .b8 97 .b8 120 .b8 95 .b8 107 .b8 101 .b8 114 .b8 110 .b8 101 .b8 108 .b8 0 .b8 1                                   // DW_AT_inline .b8 3                                   // Abbrev [3] 0x68:0x48 DW_TAG_subprogram .b64 $L__func_begin0                    // DW_AT_low_pc .b64 $L__func_end0                      // DW_AT_high_pc .b32 80                                 // DW_AT_abstract_origin .b8 4                                   // Abbrev [4] 0x7d:0x19 DW_TAG_inlined_subroutine .b32 80                                 // DW_AT_abstract_origin .b64 $L__tmp1                           // DW_AT_low_pc .b64 $L__tmp2                           // DW_AT_high_pc .b8 1                                   // DW_AT_call_file .b8 242                                 // DW_AT_call_line .b8 2 .b8 27                                  // DW_AT_call_column .b8 4                                   // Abbrev [4] 0x96:0x19 DW_TAG_inlined_subroutine .b32 80                                 // DW_AT_abstract_origin .b64 $L__tmp3                           // DW_AT_low_pc .b64 $L__tmp4                           // DW_AT_high_pc .b8 1                                   // DW_AT_call_file .b8 244                                 // DW_AT_call_line .b8 2 .b8 25                                  // DW_AT_call_column .b8 0                                   // End Of Children Mark .b8 0                                   // End Of Children Mark } .section	.debug_macinfo	{	}

## Summary

Gap between the programming model (PyTorch, Triton, PTX) and hardware => performance mysteries

Benchmarking for understanding scaling

Profiling for understanding internals of PyTorch functions (bottoms out with kernels)

Looking at PTX assembly to understand internals of CUDA kernels

5 ways to write a function: manual, PyTorch, compiled, CUDA, Triton

GeLU (element-wise), softmax (row-wise), matmul (complex aggregation)

Key principle: organize computation to minimize reads/writes

Key ideas: kernel fusion (warehouse/factory analogy), tiling (shared memory)

Automatic compilers (Triton, torch.compile) will get better over time

Horace He's blog post [[Article]](https://horace.io/brrr_intro.html)

CUDA MODE Lecture 1: how to profile CUDA kernels in PyTorch [[Video]](https://www.youtube.com/watch?v=LuhJEEJQgUM)

CUDA MODE Lecture 2: Chapters 1-3 of PPMP book [[Video]](https://www.youtube.com/watch?v=NQ-0D5Ti2dc)

CUDA MODE Lecture 3: Getting started with CUDA for Python Programmers [[Video]](https://www.youtube.com/watch?v=4sgKnKbR-WE)

CUDA MODE Lecture 4: Compute and memory basics [[Video]](https://www.youtube.com/watch?v=lTmYrKwjSOU)

CUDA MODE Lecture 8: CUDA performance checklist [[Video]](https://www.youtube.com/watch?v=SGhfUhlowB4)

HetSys Course: Lecture 1: Programming heterogenous computing systems with GPUs [[Video]](https://www.youtube.com/watch?v=8JGo2zylE80)

HetSys Course: Lecture 2: SIMD processing and GPUs [[Video]](https://www.youtube.com/watch?v=x1MA4MtO4Tc)

HetSys Course: Lecture 3: GPU Software Hierarchy [[Video]](https://www.youtube.com/watch?v=KGZ00J5MJz0)

HetSys Course: Lecture 4: GPU Memory Hierarchy [[Video]](https://www.youtube.com/watch?v=ZQKMZIP3Fzg)

HetSys Course: Lecture 5: GPU performance considerations [[Video]](https://www.youtube.com/watch?v=ODeprwr3Jho)

[[A100 GPU with NVIDIA Ampere Architecture]](https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e)

[[NVIDIA Deep Learning Performance Guide]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

[[GPU Puzzles]](https://github.com/srush/gpu-puzzles)

[[Triton Paper]](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

[[PyTorch 2.0 Acceleration]](https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26)
