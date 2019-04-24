#include <fstream>
#include <iostream>
#include "Boruvka.hpp"

using namespace std;
using namespace hdbscan;

int main() {
//  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(4);

  const uint_fast32_t n = 500000;
  const uint_fast32_t m = 40;
  Boruvka<double, uint_fast32_t> bor(n, m);
  bor.run();
  for (auto it = bor.profiler.begin(); it != bor.profiler.end(); ++it) {
    cout << it->first << ":\t\t" << it->second << '\n';
  }
//  for (uint_fast32_t i = 0; i < bor.edge_set.size(); i++) {
//    auto & edge = bor.edge_set[i];
//    cout << edge.u << " " << edge.v << "\n";
//  }
  // ofstream of;
  // of.open ("mst.txt");
  // for (auto const &e : bor.edge_set) {
  //   of << e.first / n << " " << e.first % n << " " << e.second << '\n';
  // }
  // of.close();
  cout <<  bor.output_check() << endl;
  return 0;
}