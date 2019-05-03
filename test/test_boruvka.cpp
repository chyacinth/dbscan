#include <fstream>
#include <iostream>
#include <cmath>
#include "Boruvka.hpp"
#include "SingleLinkageTree.hpp"
#include "CondensedTree.hpp"

using namespace std;
using namespace hdbscan;

using point = pair<double, double>;

vector<pair<double, double>> read_file() {
  ifstream input("/Users/claud/Desktop/CS395T-PASC/dataset/1000input.txt", ios::in);
  if (!input.is_open()) {
    std::cerr << "There was a problem opening the input file!\n";
    exit(1);
  }
  vector<pair<double, double>> points;
  double x, y;
  while(input >> x >> y) {
    points.emplace_back(make_pair(x,y));
  }
  input.close();
  return points;
}

void output_edge(vector<edge_t> edge_set) {
  ofstream output("/Users/claud/Desktop/CS395T-PASC/dataset/1000.txt", ios::out);
  if (!output.is_open()) {
    std::cerr << "There was a problem opening the input file!\n";
    exit(1);
  }
  for (auto const & edge : edge_set) {
    output << edge.u << " " << edge.v << " " << edge.w << "\n";
  }
  output.close();
}

void output_id(vector<int> ids) {
  ofstream output("/Users/claud/Desktop/CS395T-PASC/dataset/1000id.txt", ios::out);
  if (!output.is_open()) {
    std::cerr << "There was a problem opening the input file!\n";
    exit(1);
  }
  for (auto const & id : ids) {
    output << id << "\n";
  }
  output.close();
}

vector<edge_p> naive_mnn(vector<point> points, uint32_t m, uint32_t k) {
  const uint32_t n = points.size();
  vector<edge_p> ret(m*n);
  vector<vector<edge_p>> matrix(n, vector<edge_p>(n));
  for (uint32_t i = 0; i < n; i++) {
    auto & row = matrix[i];
    auto & p = points[i];
    for (uint32_t j = 0; j < n; j++) {
      auto & p1 = points[j];
      row[j] = make_pair(pow(pow(p1.first - p.first, 2.) + pow(p1.second - p.second, 2.), 0.5), j);
    }
    sort(row.begin(), row.end(), [](edge_p &e1, edge_p &e2) { return e1.first < e2.first; });
  }
  vector<double> core_d(n);
  for (uint32_t i = 0; i < n; i++) {
    core_d[i] = matrix[i][k].first;
  }
  // TODO: check possible error of updating knn on TACC
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < m; j++) {
      double now = max(max(core_d[i], core_d[matrix[i][j].second]), matrix[i][j].first);
      ret[i * m + j] = make_pair(now, matrix[i][j].second);
    }
  }
  return ret;
}

int main() {
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(4);

  auto points = read_file();
  const uint32_t n = points.size();
  const uint32_t k = 10;
  const uint32_t m = 7;

  auto mat = naive_mnn(points, m, k);

  Boruvka<double, uint32_t> bor(n, m, mat);
  bor.run();
  for (auto it = bor.profiler.begin(); it != bor.profiler.end(); ++it) {
    cout << it->first << ":\t\t" << it->second << '\n';
  }
  cout <<  bor.output_check() << endl;
//  output_edge(bor.edge_set);

  hdbscan::SingleLinkageTree<float, int> slt{bor.edge_set, k};
  hdbscan::CondensedTree<float, int> ct{slt};
  ct.print(false, true);
  vector<int> point_cluster = ct.get_point_cluster();
  point_cluster.pop_back();
  output_id(point_cluster);

  return 0;
}