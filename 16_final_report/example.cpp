#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#include <mpi.h>

#pragma clang diagnostic pop

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#ifdef USE_SIMD
#include <immintrin.h>
#endif  // USE_SIMD

using namespace std;

#if 0
// clang-format off
int main(int argc, char** argv) {
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/size);
  vector<float> subB(N*N/size);
  vector<float> subC(N*N/size, 0);
  vector<float> recv(N*N/size);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        for (int k=0; k<N; k++)
          subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    MPI_Request request[2];
    MPI_Isend(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&recv[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
    for (int i=0; i<N*N/size; i++)
      subB[i] = recv[i];
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();
}
// clang-format on
#endif

// A: m x k
// B: k x n
// C: m x n
//
static void my_sub_matmul(const float* A, const float* B, float* C, const int m,
                          const int n, const int k, const int stride_C,
                          const int offset_C) {
  const int kc = 512;
  const int nc = 64;
  const int mc = 256;
  const int nr = 64;
  const int mr = 32;

  float Bc[kc * nc];
  float Ac[mc * kc], Cc[mc * nc];

#pragma omp parallel for collapse(2) private(Bc) private(Ac) private(Cc)
  for (int jc = 0; jc < n; jc += nc) {
    for (int pc = 0; pc < k; pc += kc) {
      const int _kc = std::min(kc, k - pc);
      const int _nc = std::min(nc, n - jc);

      for (int p = 0; p < _kc; ++p) {
        int j = 0;
#ifdef USE_SIMD
        for (; j + 7 < _nc; j += 8) {
          __m256 Bvec = _mm256_loadu_ps(B + (p + pc) * n + j + jc);
          _mm256_storeu_ps(Bc + p * nc + j, Bvec);
        }
#endif
        for (; j < _nc; ++j) {
          Bc[p * nc + j] = B[(p + pc) * n + j + jc];
        }
      }

      for (int ic = 0; ic < m; ic += mc) {
        const int _mc = std::min(mc, m - ic);
        for (int i = 0; i < _mc; ++i) {
          int p = 0;
#ifdef USE_SIMD
          for (; p + 7 < _kc; p += 8) {
            __m256 Avec = _mm256_loadu_ps(A + (i + ic) * k + p + pc);
            _mm256_storeu_ps(Ac + i * kc + p, Avec);
          }
#endif
          for (; p < _kc; ++p) {
            Ac[i * kc + p] = A[(i + ic) * k + p + pc];
          }
          int j = 0;
#ifdef USE_SIMD
          for (; j + 7 < _nc; j += 8) {
            _mm256_storeu_ps(Cc + i * nc + j, _mm256_set1_ps(0.f));
          }
#endif
          for (; j < _nc; ++j) {
            Cc[i * nc + j] = 0;
          }
        }
        for (int jr = 0; jr < nc; jr += nr) {
          for (int ir = 0; ir < mc; ir += mr) {
            for (int kr = 0; kr < _kc; ++kr) {
              for (int i = ir; i < std::min(ir + mr, _mc); ++i) {
#ifdef USE_SIMD
                __m256 Avec = _mm256_broadcast_ss(Ac + i * kc + kr);
#endif

                int j = jr;
#ifdef USE_SIMD
                for (; j + 7 < std::min(jr + nr, _nc); j += 8) {
                  __m256 Bvec = _mm256_loadu_ps(Bc + kr * nc + j);
                  __m256 Cvec = _mm256_loadu_ps(Cc + i * nc + j);
                  Cvec        = _mm256_fmadd_ps(Avec, Bvec, Cvec);
                  _mm256_storeu_ps(Cc + i * nc + j, Cvec);
                }
#endif
                for (; j < std::min(jr + nr, _nc); ++j) {
                  Cc[i * nc + j] += Ac[i * kc + kr] * Bc[kr * nc + j];
                }
              }
            }
          }
        }
        for (int i = 0; i < _mc; ++i) {
          int j = 0;
#ifdef USE_SIMD
          for (; j + 7 < _nc; j += 8) {
            __m256 Ccvec = _mm256_loadu_ps(Cc + i * nc + j);
            __m256 Cvec =
                _mm256_loadu_ps(C + (i + ic) * stride_C + j + jc + offset_C);
            Cvec = _mm256_add_ps(Ccvec, Cvec);
            _mm256_storeu_ps(C + (i + ic) * stride_C + j + jc + offset_C, Cvec);
          }
#endif
          for (; j < _nc; ++j) {
            C[(i + ic) * stride_C + j + jc + offset_C] += Cc[i * nc + j];
          }
        }
      }
    }
  }

}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 4000;

  vector<float> A(N * N);
  vector<float> B(N * N);
  vector<float> C(N * N, 0);
  vector<float> subA(N * N / size);
  unique_ptr<float[]> subB = std::make_unique<float[]>(N * N / size);
  vector<float> subC(N * N / size, 0);
  unique_ptr<float[]> recv = std::make_unique<float[]>(N * N / size);

  if (rank == 0) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] = float(drand48());
        B[N * i + j] = float(drand48());
      }
    }
  }
  MPI_Bcast(A.data(), N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B.data(), N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int offset = N / size * rank;
  for (int i = 0; i < N / size; i++)
    for (int j = 0; j < N; j++) subA[N * i + j] = A[N * (i + offset) + j];
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N / size; j++)
      subB[N / size * i + j] = B[N * i + j + offset];
  int recv_from = (rank + 1) % size;
  int send_to   = (rank - 1 + size) % size;

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

  // Checks error
  if (rank == 0) {
#pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++) C[N * i + j] -= A[N * i + k] * B[N * k + j];
    double err = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) err += double(fabs(C[size_t(N * i + j)]));

    double time = comp_time + comm_time;
    printf("N    : %d\n", N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
    printf("error: %lf\n", err / N / N);
  }
  MPI_Finalize();
  return 0;
}

