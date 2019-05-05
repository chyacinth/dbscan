/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *
 *  Copyright (C) 2014-2018, The University of Texas at Austin
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/

#include <stdint.h>
#include <limits.h>
#include <algorithm>

/** Use MPI-GOFMM templates. */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
#include "Distributed_boruvka.hpp"
#include "SingleLinkageTree.hpp"
#include "CondensedTree.hpp"

#define DIMENSION 2

#if SIZE_MAX == UCHAR_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

using namespace std;
using namespace hmlp;

/**
 *  @brief In this example, we explain how you can compute
 *         approximate all-nearest neighbors (ANN) using MPIGOFMM.
 */
 double dist(double* a, double* b, size_t d) {
   double result = 0;
   for (int i = 0; i < d; ++i) {
     result += (a[i] - b[i]) * (a[i] - b[i]);
   }
   return result;
 }
int main( int argc, char *argv[] )
{
  try
  {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(4); // Use 4 threads for all consecutive parallel regions

    /** Use float as data type. */
    using T = double;
    size_t core_k = 10;
    /** [Required] Problem size. */
    size_t n = 2000;
    /** Maximum leaf node size (not used in neighbor search). */
    size_t m = 128;
    /** [Required] Number of nearest neighbors. */
    size_t k = 11;
    /** Maximum off-diagonal rank (not used in neighbor search). */
    size_t s = 128;
    /** Approximation tolerance (not used in neighbor search). */
    T stol = 1E-5;
    /** The amount of direct evaluation (not used in neighbor search). */
    T budget = 0.01;

    int inspect = 1;
    /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
    int  provided = 0;
    mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    if ( provided != MPI_THREAD_MULTIPLE ) exit( 1 );
    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
    /** [Step#0] HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( &argc, &argv, CommGOFMM ) );

    /** Here neighbors1 is distributed in DistData<STAR, CBLK, T> over CommGOFMM. */
    int rank; mpi::Comm_rank( CommGOFMM, &rank );
    int size; mpi::Comm_size( CommGOFMM, &size );

    /** [Step#1] Create a configuration for kernel matrices. */
    gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
    /** [Step#2] Create a distributed Gaussian kernel matrix with random 6D data. */
    DistData<STAR, CBLK, T> X( DIMENSION, n, CommGOFMM ); //X.randn();

    fstream file("/home/parallels/workspace/dbscan/data/2000input.txt");
    double x = 0;
    double y = 0;
    int cnt = 0;
    while (file >> x >> y) {
      if (cnt % size == rank) {
        int local_cnt = (cnt - rank) / size;
        X.data()[DIMENSION * local_cnt] = x;
        X.data()[DIMENSION * local_cnt + 1] = y;
      }
      ++cnt;
    }
    //cout << "cnt == " << cnt << endl;



    DistKernelMatrix<T, T> K2( X, CommGOFMM );
    /** [Step#3] Create a distributed randomized splitter. */
    mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter2( K2 );
    /** [Step#4] Perform the iterative neighbor search. */
    auto neighbors2 = mpigofmm::FindNeighbors( K2, rkdtsplitter2, config2, CommGOFMM );
    size_t row = neighbors2.row_owned();
    size_t col_owned = neighbors2.col_owned();
    size_t row_owned = neighbors2.row_owned();

    /*for (size_t j = 0; j < neighbors2.col_owned(); ++j) {
      for (size_t i = 0; i < neighbors2.row_owned(); ++i) {
        neighbors2.data()[j * row + i].first = sqrt(neighbors2.data()[j * row + i].first);
      }
    }*/

    /*if (rank == inspect) {
      cout << "print neighbour2" << endl;
      cout << neighbors2(1, inspect).second << " " << neighbors2(1, size + inspect).second << endl;
    }*/

    struct pr {
      double first = 0;
      size_t second = 0;
      double coord[DIMENSION]{};
    };

    //using pr = pair<T, size_t>;

    // init MPI
    MPI_Datatype etype;
    int block_length[3] = {1, 1, static_cast<int>(DIMENSION)};
    MPI_Aint displacements[3] = {offsetof(pr, first), offsetof(pr, second), offsetof(pr, coord)};
    MPI_Datatype types[3] = {MPI_DOUBLE, my_MPI_SIZE_T, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_length, displacements, types, &etype);
    MPI_Type_commit(&etype);


    /*if (rank == inspect) {
      cout << endl;
      for (size_t i = 0; i < 10; ++i) {
        cout << "(" << neighbors2(core_k, rank + i * size).first << ", " << neighbors2(core_k, rank + i * size).second << ")  ";
      }
      cout << endl;
    }*/


    vector<pr> data_send (neighbors2.col_owned());
    #pragma omp parallel for
    for (size_t i = 0; i < col_owned; ++i) {
      size_t actual_i = rank + i * size;
      data_send[i].first = neighbors2(core_k, actual_i).first;
      data_send[i].second = neighbors2(core_k, actual_i).second;
      //data_send[i].coord = new double[d];
      for (size_t j = 0; j < DIMENSION; ++j) {
        data_send[i].coord[j] = X(j, actual_i);
      }
    }

    /*if (rank == 0) {
      cout << "original X in process 0: col 0 ~ 30: " << endl;
      for (int i = 0; i < 30; ++i) {
        if (i % size == 0)
          cout << "(" << X(0, i) << ", " << X(1, i) << ", " << X(2, i) << ")  ";
      }
      cout << endl;
    }*/

    //TODO: change size?
    vector<pr> recv_data(n);
    //cout << "data_send.size()" << data_send.size() << endl;
    //cout << "recv_data.size()" << recv_data.size() << endl;
    MPI_Allgather(data_send.data(), static_cast<int>(data_send.size()), etype, recv_data.data(),
                  static_cast<int>(n / size), etype, CommGOFMM);

    /*if (rank == inspect) {
      cout << "received recv_data: column 0 ~ 15 " << endl;
      cout << recv_data[500].first << " " << recv_data[500].second << endl;
      cout << endl;
    }*/

    vector<int> sums(size);
    for (int i = 0; i < size; ++i) {
      if (i > 0) {
        sums[i] = sums[i - 1];
        sums[i] += (n - i) / size + 1;
      } else {
        sums[i] = 0;
      }
    }
    /*if (rank == inspect) {
      for (int i = 0; i < size; ++i) {
        cout << sums[i] << " ";
      }
      cout << endl;
    }*/

    /*if (rank == inspect) {
      size_t jj = 0;
      cout << "final test: " << endl;
      size_t i = 7;
      size_t j = 367;
      size_t actual_j = rank + j * size;
      cout << "actual j: " << actual_j << endl;
      jj = neighbors2(i, actual_j).second;

      /*cout << neighbors2.data()[row * j + i].first << " " << neighbors2.data()[row * j + i].second << endl;
      cout << neighbors2(i, actual_j).first << " " << neighbors2(i, actual_j).second << endl;
      cout << endl;

      cout << "1: " << neighbors2(core_k, actual_j).first << " " << neighbors2(core_k, actual_j).second << endl;*/

      /*for (int i = 0; i < DIMENSION; ++i) {
        cout << X(i, actual_j) << " ";
      }*/

      //cout << X(actual_j, neighbors2.data()[row * j + i].second) << " ";

      /*cout << endl;
      for (int jj = 0; jj < n; ++jj) {
        if (jj != recv_data[sums[jj % size] + (jj - (jj % size)) / size].second) {
          cout << "wrong!!";
        }
      }
      cout << endl;
    }*/

    /*if (rank == 704 % size) {
      cout << "2: " << neighbors2(core_k, 704).first << " " << neighbors2(core_k, 704).second << endl;
      for (int i = 0; i < d; ++i) {
        cout << X(i, 704) << " ";
      }
      cout << endl;
    }*/

    DistData<STAR, CBLK, pair<T, size_t>> neighbors_final(neighbors2.row() - 1, neighbors2.col(), CommGOFMM);

    //cout << "row is: " << row << endl;

    --row;

    vector<double> tc(DIMENSION);
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < col_owned; ++j) {
      for (size_t i = 1; i < row_owned; ++i) {
        size_t actual_j = rank + j * size;
        size_t other_j = neighbors2(i, actual_j).second;
        for (size_t l = 0; l < DIMENSION; ++l) {
          tc[l] = X(l, actual_j);
        }
        neighbors_final.data()[row * j + i - 1].first = std::max({
          neighbors2(core_k, actual_j).first,
          dist(recv_data[sums[other_j % size] + ((other_j - (other_j % size)) / size)].coord, tc.data(), DIMENSION),
          recv_data[sums[other_j % size] + ((other_j - (other_j % size)) / size)].first
          });
        neighbors_final.data()[row * j + i - 1].second = other_j;
      }
    }

    //cout << "final row & col: " << neighbors_final.row() << " " << neighbors_final.col() << endl;
    //cout << "boruvka begins" << endl;

    hdbscan::Distributed_Boruvka<double, int32_t> boruvka(neighbors_final, CommGOFMM);
    boruvka.run();
    //cout << "boruvka ends" << endl;

    if (rank == 0) {
      cout << boruvka.edge_set.size() << endl;
      cout << "edge_set: " << endl;
      size_t edge_size = boruvka.edge_set.size();
      #pragma omp parallel for
      for (int i = 0; i < edge_size; ++i) {
        boruvka.edge_set[i].w = sqrt(boruvka.edge_set[i].w);
      }
      hdbscan::SingleLinkageTree<double, int32_t> slt{boruvka.edge_set, 10};
      hdbscan::CondensedTree<double, int32_t> ct{slt};
      ct.print();
    }

    /** [Step#5] HMLP API call to terminate the runtime. */
    HANDLE_ERROR( hmlp_finalize() );
    /** Finalize Message Passing Interface. */
    mpi::Finalize();

  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}; /** end main() */
