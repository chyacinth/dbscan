#include <iostream>
#include <chrono> 
#include <fstream>

#include "SingleLinkageTree.hpp"
#include "CondensedTree.hpp"
#include "Boruvka.hpp"
//#include "Distributed_boruvka.hpp"

using namespace std;
using namespace std::chrono;

int main() {
  try
  {
    using T = float;
    using U = int32_t;

    ios::sync_with_stdio(false);

    ifstream file("/Users/hyacinth/workspace/dbscan/data/1000.txt");
    vector<hdbscan::SingleLinkageTree<T, U >::edge_t> mst{};
    U id1;
    U id2;
    T distance;
    while (file >> id1 >> id2 >> distance) {
      mst.emplace_back(id1, id2, distance);
    }    

    //hdbscan::Boruvka<T, U> boruvka{1500};
    auto start = high_resolution_clock::now();
    hdbscan::SingleLinkageTree<T, U> slt{mst, 30};
    hdbscan::CondensedTree<T, U> ct{slt};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);     
    //ct.print();
    cout << "Time taken by function: "
      << duration.count() << " milliseconds" << endl; 
    ct.print(true, true);
    auto pc = ct.get_point_cluster();
    cout << pc.size() << endl;
    for (size_t i = 0; i < pc.size(); ++i)
      cout << i << " " << pc[i] << endl;
  }  
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}