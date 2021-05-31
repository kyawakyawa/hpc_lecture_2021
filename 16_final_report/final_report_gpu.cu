#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#include <mpi.h>

#pragma clang diagnostic pop

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace std;

__global__ void matmul(float *A, float *B, float *C, int k) {
  int m = gridDim.y;
  int n = blockDim.x * gridDim.x;

  int i = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
  for (int ks=0; ks<k; ks+=blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[k * i + ks + threadIdx.x];
    __syncthreads();
    for (int _k = ks; _k < ks + blockDim.x; _k++) {
      sum += A_s[_k - ks] * B[n * _k + j];
    }
  }
  C[n * i + j] = sum;
}

void my_sub_matmul(float *A, float *B, float *C, const int m, const int n,
             const int k, const int stride_C, const int offset_C) {
  float* A_gpu;
  float* B_gpu;
  float* C_gpu;

  std::unique_ptr<float[]> tmpC(new float[m * n]);
  // for (int i = 0; i < m * n; ++i) tmpC[i] = 0.f;

  cudaMalloc((void**)&A_gpu, m * k * sizeof(float));
  cudaMalloc((void**)&B_gpu, k * n * sizeof(float));
  cudaMalloc((void**)&C_gpu, m * n * sizeof(float));


  cudaMemcpy(A_gpu, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(C_gpu, tmpC.get(), m * n * sizeof(float), cudaMemcpyHostToDevice);
   cudaDeviceSynchronize();

  //////////////////////////
   int M = std::min(1024, n);
   dim3 grid(n / M, m);
   matmul<<<grid,M, M*sizeof(float)>>>(A_gpu, B_gpu, C_gpu, k);
   cudaDeviceSynchronize();
  //////////////////////////

  cudaMemcpy(tmpC.get(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);

   cudaDeviceSynchronize();

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);

// #pragma omp parallel for collapse(2)
  for (int i = 0;i < m;++i) {
    for (int j = 0;j < n;++j) {
      C[i * stride_C + j + offset_C] = tmpC[i * n + j];
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 4096;

  vector<float> A(N * N);
  vector<float> B(N * N);
  vector<float> C(N * N, 0);
  vector<float> subA(N * N / size);
  unique_ptr<float[]> subB(new float[N * N / size]);
  vector<float> subC(N * N / size, 0);
  unique_ptr<float[]> recv(new float[N * N / size]);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[N * i + j] = float(drand48());
      B[N * i + j] = float(drand48());
    }
  }

  int offset = N / size * rank;
  for (int i = 0; i < N / size; i++)
    for (int j = 0; j < N; j++) subA[N * i + j] = A[N * (i + offset) + j];
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N / size; j++)
      subB[N / size * i + j] = B[N * i + j + offset];
  int recv_from = (rank + 1) % size;
  int send_to   = (rank - 1 + size) % size;

  MPI_Barrier(MPI_COMM_WORLD);

  double comp_time = 0, comm_time = 0;
  for (int irank = 0; irank < size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset   = N / size * ((rank + irank) % size);
    my_sub_matmul(subA.data(), subB.get(), subC.data(), N / size, N / size, N,
                  N, offset);
    const auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    if (irank < size - 1) {
      MPI_Request request[2];
      MPI_Isend(subB.get(), N * N / size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD,
                &request[0]);
      MPI_Irecv(recv.get(), N * N / size, MPI_FLOAT, recv_from, 0,
                MPI_COMM_WORLD, &request[1]);
      MPI_Waitall(2, request, MPI_STATUS_IGNORE);
      subB.swap(recv);
    }
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  const auto tic = chrono::steady_clock::now();
  MPI_Allgather(&subC[0], N * N / size, MPI_FLOAT, &C[0], N * N / size,
                MPI_FLOAT, MPI_COMM_WORLD);
  const auto toc = chrono::steady_clock::now();
  comm_time += chrono::duration<double>(toc - tic).count();

  if (rank == 0) {
    double time = comp_time + comm_time;
    printf("N    : %d\n", N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
  }

  // Checks error
#pragma omp parallel for
  for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++) C[N * i + j] -= A[N * i + k] * B[N * k + j];
  double err = 0;
#pragma omp parallel for
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) err += double(fabs(C[size_t(N * i + j)]));

  if (rank == 0) {
    printf("error: %lf\n", err / N / N);
  }
  MPI_Finalize();
  return 0;
}
