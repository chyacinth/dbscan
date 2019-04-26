#ifndef DISTRIBUTED_BORUVKA_H_
#define DISTRIBUTED_BORUVKA_H_

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
//#define NDEBUG
#include <assert.h>
#include <chrono>
#include <limits>
#include <map>
#include <omp.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include "mpi.h"

#define ROUND_UP(x, s) (((x) + ((s)-1)) & -(s))
#define par_for _Pragma("omp parallel for") for
// #define par_for for

namespace hdbscan {
template <typename T, typename U> class Distributed_boruvka {
  using edge_t = std::pair<U, T>;
 public:
  explicit Distributed_boruvka(U n_) : n(n_), k(n_) {
    U rn = ROUND_UP(n, block_size);
    edges = std::vector<std::vector<edge_t>>(rn, std::vector<edge_t>(rn));
    edges_tmp = std::vector<std::vector<edge_t>>(rn, std::vector<edge_t>(rn));
    // random_device rd;
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> rand_gen(1.0, 100.0);
    for (U i = 0; i < n; i++)
      for (U j = 0; j < n; j++)
        if (i < j)
          edges[i][j] = std::make_pair(i * n + j, rand_gen(gen));
        else if (i > j)
          edges[i][j] = std::make_pair(i * n + j, edges[j][i].second);
        else
          edges[i][j] = std::make_pair(i * n + j, inf);
    // label stores the roots of the components
    rep = std::vector<U>(n);
    std::iota(rep.begin(), rep.end(), 0);
    label = std::vector<U>(n);
    std::iota(label.begin(), label.end(), 0);
    end_ind = std::vector<U>(n);
  }

  void run() {
    auto start = now_time();
    reset_clock();
    while (true) {
      // cout << k << "\n";
      /* find min edges of each compoment : O(k^2/p) */
      find_min();
      profile("1.find_min");

      /* relabel reps according to min edges */
      // just leave it sequential?
      U count_relabel = relabel();
      new_k = k - count_relabel;
      profile("2.relabel");
      if (new_k == 1)
        break;

      /* shrink to rooted star */
      shrink();
      profile("3.shrink");
      // pointer_jump();

      /* compact graph */
      // sort label
      std::vector<U> indices = argsort(label, k);
      sort(label.begin(), label.begin() + k);
      profile("4.sort_label");

      // rows
      compact_rows(indices);
      profile("5.compact_rows");
      // transpose weight and edge
      transpose();
      profile("6.transpose");
      // columns after transposion
      compact_columns(indices);
      profile("7.compact_columns");

      // edges.resize(new_k);
      // edges_tmp.resize(new_k);
      // profile("8.resize");
      edges.swap(edges_tmp);
      // compact label
      compact_label();
      // label.resize(new_k);
      k = new_k;
      profile("8.shrink_label");
    }
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(now_time() - start).count() /
        1000000.
              << "\n";
    std::cout << edge_set.size() << std::endl;
    // sort edges
    sort(edge_set.begin(), edge_set.end(),
         [](edge_t &e1, edge_t &e2) { return e1.second < e2.second; });
  }

 private:
  constexpr static auto now_time = std::chrono::high_resolution_clock::now;

  static const U block_size = 32;
  static const U inner_block_size = 4;

  T inf = std::numeric_limits<T>::infinity();
  const U n;
  U k;
  U new_k;
  std::vector<std::vector<edge_t>> edges;
  std::vector<std::vector<edge_t>> edges_tmp;
  std::vector<edge_t> edge_set;
  std::vector<U> rep, label;
  std::vector<U> end_ind;
  std::map<std::string, float> profiler;
  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

  inline U argmin(std::vector<edge_t> const &v, const U size) {
    auto result = min_element(v.begin(), v.begin() + size,
                              [](edge_t e1, edge_t e2) { return e1.second < e2.second; });
    return distance(v.begin(), result);
  }
  inline std::vector<U> argsort(const std::vector<U> &v, const U size) {
    std::vector<U> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](U i1, U i2) { return v[i1] < v[i2]; });
    return idx;
  }
  inline void reset_clock() { timestamp = now_time(); }
  inline void profile(std::string name) {
    profiler[name] +=
        std::chrono::duration_cast<std::chrono::microseconds>(now_time() - timestamp).count() /
        1000000.;
    timestamp = now_time();
  }

  inline void find_min() { par_for (U i = 0; i < k; i++) end_ind[label[i]] = argmin(edges[i], k); }

  inline U relabel() {
    U count_relabel = 0;
    for (U i = 0; i < k; i++) {
      U this_end_rep = label[i];
      U other_end_ind = end_ind[this_end_rep];
      U other_end_rep = label[other_end_ind];
      U third_end_rep = label[end_ind[other_end_rep]];
      if (third_end_rep != this_end_rep || this_end_rep > other_end_rep) {
        rep[this_end_rep] = other_end_rep;
        edge_set.push_back(edges[i][other_end_ind]);
        count_relabel++;
      }
    }
    return count_relabel;
  }

  inline void shrink() {
    for (U i = 0; i < k; i++) {
      U this_label = label[i];
      U root = this_label;
      while (root != rep[root])
        root = rep[root];
      U now = this_label;
      while (now != root) {
        now = rep[now];
        rep[now] = root;
      }
      rep[this_label] = root;
      label[i] = root;
    }
  }
  inline void pointer_jump() {
    // pointer jump: seems slightly slower, even in parallel
    par_for(U i = 0; i < k; i++) {
      U this_label = label[i];
      while (rep[this_label] != rep[rep[this_label]])
        rep[this_label] = rep[rep[this_label]];
      label[i] = rep[this_label];
    }
  }

  inline void compact_rows(std::vector<U> const &indices) {
    par_for(U i = 0; i < k; i++) {
      auto &edge_row = edges[i];
      auto &tmp_row = edges_tmp[i];
      int ind = -1;
      int last = -1;
      for (U j = 0; j < k; j++) {
        edge_t now_edge = edge_row[indices[j]];
        if (label[j] != last) {
          last = label[j];
          ind++;
          tmp_row[ind] = now_edge;
        } else if (tmp_row[ind].second > now_edge.second) {
          tmp_row[ind] = now_edge;
        }
      }
    }
  }

  inline void inner_trans(U const i, U const j) {
    for (U i2 = 0; i2 < block_size; i2 += inner_block_size)
      for (U j2 = 0; j2 < block_size; j2 += inner_block_size)
        for (U i1 = 0; i1 < inner_block_size; i1++)
          for (U j1 = 0; j1 < inner_block_size; j1++)
            edges[i2 + i1 + i][j2 + j1 + j] = edges_tmp[j2 + j1 + j][i2 + i1 + i];
  }
  inline void transpose() {
    // need fast SSE + openmp treatment
    par_for(U i = 0; i < new_k; i += block_size) {
      for (U j = 0; j < k; j += block_size) {
        // int max_i2 = i + block_size < new_k ? i + block_size : new_k;
        // int max_j2 = j + block_size < k ? j + block_size : k;
        // no need to swap u / v. they are not directly referred in
        // this code
        inner_trans(i, j);
        // edges[i][j] = edges_tmp[j][i];
      }
    }
  }

  inline void compact_columns(std::vector<U> const &indices) {
    par_for(U i = 0; i < new_k; i += 1) {
      auto &edge_row = edges[i];
      auto &tmp_row = edges_tmp[i];
      int ind = -1;
      int last = -1;
      for (U j = 0; j < k; j++) {
        edge_t now_edge = edge_row[indices[j]];
        if (label[j] != last) {
          last = label[j];
          ind++;
          if (ind == i)
            tmp_row[ind].second = inf;
          else
            tmp_row[ind] = now_edge;
        } else {
          T w = now_edge.second;
          auto last_w = tmp_row[ind].second;
          if (last_w != inf && last_w > w)
            tmp_row[ind] = now_edge;
        }
      }
      // necessary?
      // edge_row.resize(new_k);
      // tmp_row.resize(new_k);
    }
  }

  inline void compact_label() {
    U ind = 0;
    U last = label[0];
    for (U i = 1; i < k; i++) {
      if (label[i] != last) {
        ind++;
        last = label[i];
        label[ind] = last;
      }
    }
    assert(ind + 1 == new_k);
  }

  int output_check() {
    std::vector<U> rep_check(n);
    std::iota(rep_check.begin(), rep_check.end(), 0);
    for (auto const &edge : edge_set) {
      U u = edge.first / n;
      U v = edge.first % n;
      // cout << u << ",";
      auto root = u;
      while (root != rep_check[root])
        root = rep_check[root];
      auto now = u;
      while (now != root) {
        auto tmp = now;
        rep[now] = root;
        now = rep[now];
      }

      root = v;
      while (root != rep_check[root])
        root = rep_check[root];
      now = v;
      while (now != root) {
        auto tmp = now;
        rep[now] = root;
        now = rep[now];
      }
      if (rep_check[u] == rep_check[v])
        return -1;
      rep_check[u] = v;
    }
    return 0;
  }
};
} // namespace hdbscan
#endif
