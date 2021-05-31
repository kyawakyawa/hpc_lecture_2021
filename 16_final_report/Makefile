build:
	mpicxx example.cpp -fopenmp -march=native -O3 -DUSE_SIMD
	#OMPI_CC=clang-10 OMPI_CXX=clang++-10 mpicxx example.cpp -fopenmp -march=native -O3 -DUSE_SIMD -Weverything -Wno-sign-conversion -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-zero-as-null-pointer-constant 
	#OMPI_CC=gcc-10 OMPI_CXX=g++-10 mpicxx example.cpp -march=native -O3 -fopenmp 
run:
	OMP_NUM_THREADS=4 mpirun -np 4 ./a.out
