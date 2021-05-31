build:
	mpicxx final_report_cpu.cpp -fopenmp -march=native -O3 -DUSE_SIMD -std=c++11
	#OMPI_CC=clang-10 OMPI_CXX=clang++-10 mpicxx final_report_cpu.cpp -fopenmp -march=native -O3 -DUSE_SIMD -Weverything -Wno-sign-conversion -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-zero-as-null-pointer-constant -std=c++11
	#OMPI_CC=gcc-10 OMPI_CXX=g++-10 mpicxx final_report_cpu.cpp -march=native -O3 -fopenmp -DUSE_SIMD -std=c++11
	nvcc final_report_gpu.cu -lmpi  -arch=sm_60 -o gpu -O3 -Xcompiler "-O3 -fopenmp -march=native"
run:
	OMP_NUM_THREADS=2 mpirun -np 8 ./a.out
gpu_run:
	OMP_NUM_THREADS=2 mpirun -np 8 ./gpu
