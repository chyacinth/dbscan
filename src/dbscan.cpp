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

/** Use GOFMM templates. */
#include <gofmm.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
#include "SingleLinkageTree.hpp"

using namespace std;
using namespace hmlp;

template<typename T>
vector<vector<T>> find_nearest_neighbours()
{  
  /** [Required] Problem size. */
  size_t n = 50;
  /** Maximum leaf node size (not used in neighbor search). */
  size_t m = 128;
  /** [Required] Number of nearest neighbors. */
  size_t k = 7;
  /** Maximum off-diagonal rank (not used in neighbor search). */
  size_t s = 128;
  /** Approximation tolerance (not used in neighbor search). */
  T stol = 1E-5;
  /** The amount of direct evaluation (not used in neighbor search). */
  T budget = 0.01;

  /** [Step#1] Create a configuration for kernel matrices. */
  gofmm::Configuration<T> config( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );

  /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
  size_t d = 2;
  Data<T> X( d, n ); X.randn();
  KernelMatrix<T> K( X );
  cout << "Number of rows: " << K.row() << " number of columns: " << K.col() << endl;
  cout << "K(0,0) " << K( 0, 0 ) << " K(1,2) " << K( 1, 2 ) << endl;

  /** [Step#3] Create a randomized splitter. */
  gofmm::randomsplit<KernelMatrix<T>, 2, T> rkdtsplitter( K );

  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors = gofmm::FindNeighbors( K, rkdtsplitter, config );
  cout << "Number of neighbours: " << neighbors.row() << " number of queries: " << neighbors.col() << endl;
  for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
    printf( "[%E,%5lu]\n", neighbors( i, 0 ).first, neighbors( i, 0 ).second );    

  // construct mutal reachability matrix
  vector<vector<T>> m_reach = vector<vector<T>>(n, vector<T>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      m_reach[i][j] = std::max({neighbors(k, i).first, neighbors(k, j).first, K(i, j)});
    }
  }

  return m_reach;

};

int main( int argc, char *argv[] ) {
  try
  {
    // Use float as data type.
    using T = float;
    
    // HMLP API call to initialize the runtime.
    HANDLE_ERROR( hmlp_init( &argc, &argv ) );

    //auto m_reach = find_nearest_neighbours<T>();

    vector<tuple<int, int, float>> mst{};
    mst.push_back(make_tuple(0, 1, 1.5));
    mst.push_back(make_tuple(1, 2, 2.4));
    mst.push_back(make_tuple(0, 2, 3.1));
    mst.push_back(make_tuple(3, 0, 4.5));


    // [Step#5] HMLP API call to terminate the runtime.
    HANDLE_ERROR( hmlp_finalize() );
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}