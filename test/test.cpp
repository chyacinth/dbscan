#include <iostream>
#include <chrono> 
#include <fstream>

#include "SingleLinkageTree.hpp"
#include "CondensedTree.hpp"
#include "Boruvka.hpp"
#include "Distributed_boruvka.hpp"

using namespace std;
using namespace std::chrono;

int main() {
  try
  {
    using T = float;
    using U = int;

    ios::sync_with_stdio(false);

    ifstream file("/Users/hyacinth/workspace/dbscan/data/mst.txt");
    vector<tuple<U, U, T>> mst{};
    U id1;
    U id2;
    T distance;
    while (file >> id1 >> id2 >> distance) {
      mst.emplace_back(make_tuple(id1, id2, distance));
    }    

    hdbscan::Boruvka<T, U> boruvka{1500};        
    auto start = high_resolution_clock::now();
    hdbscan::SingleLinkageTree<T, U> slt{mst, 2};
    hdbscan::CondensedTree<T, U> ct{slt};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);     
    //ct.print();
    cout << "Time taken by function: "
      << duration.count() << " milliseconds" << endl; 
    ct.print();
  }  
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}