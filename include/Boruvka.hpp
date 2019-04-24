#ifndef BORUVKA_H_
#define BORUVKA_H_
//#define NDEBUG

// backup: sparse

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cassert>
#include <chrono>
#include <limits>
#include <map>
#include <omp.h>
#include <string>
#include <ctime>
#include <atomic>
#include <unordered_set>

#define ROUND_UP(x, s) (((x) + ((s)-1)) & -(s))
#define par_for _Pragma("omp parallel for") for
#define par_for_256 _Pragma("omp parallel for schedule (static,256)") for
// #define par_for for

namespace hdbscan {
template <typename T, typename U> class Boruvka {
  struct edge_t {
    U u;
    U v;
    T w;
    public:
    edge_t() {};
    edge_t(U u_, U v_, T w_) : u(u_), v(v_), w(w_) {};
  };
  constexpr static auto now_time = std::chrono::high_resolution_clock::now;

//  inline U argmin(std::vector<edge_t> const &v, const U size) {
//    auto result = std::min_element(v.begin(), v.begin() + size,
//                              [](edge_t e1, edge_t e2) { return e1.w < e2.w; });
//    return distance(v.begin(), result);
//  }

//  inline std::vector<U> argsort(const std::vector<U> &v, const U size) {
//    std::vector<U> idx(size);
//    std::iota(idx.begin(), idx.end(), 0);
//    std::sort(idx.begin(), idx.end(), [&v](U i1, U i2) { return v[i1] < v[i2]; });
//    return idx;
//  }

  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
  inline void reset_clock() { timestamp = now_time(); }
  inline void profile(std::string name, bool p=false) {
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(now_time() - timestamp).count() /
        1000000.;
    if (p) std::cout << vertex_left << "\t" << name << "\t" << time << std::endl;
    profiler[name] += time;
    timestamp = now_time();
  }

public:
  T inf = std::numeric_limits<T>::infinity();
  const U n, m;
  U vertex_left;
  std::vector<std::vector<edge_t>> edges;
  std::vector<edge_t> edge_set;
  std::vector<U> rep;
  std::vector<edge_t> end_ind;
//  std::vector<U> length;

  std::map<std::string, double> profiler;

  Boruvka(U n_, U m_) : n(n_), m(m_), vertex_left(n_) {
    edges = std::vector<std::vector<edge_t>>(n, std::vector<edge_t>(m));
    // random_device rd;
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> weight_gen(1.0, 100.0);
    std::uniform_int_distribution<U> id_gen(0, n-1);
    for (U i = 0; i < n; i++) {
      std::unordered_set<U> gen_set;
      for (U j = 0; j < m; j++) {
        U generated = id_gen(gen);
        while (generated == i or gen_set.find(generated) != gen_set.end()) {
          generated = id_gen(gen);
        }
        edges[i][j] = edge_t(i, generated, weight_gen(gen));
      }
    }
    end_ind = std::vector<edge_t>(n, edge_t(0,0,inf));
    // label stores the roots of the components
    rep = std::vector<U>(n);
    std::iota(rep.begin(), rep.end(), 0);
//    length = std::vector<U>(n, m);
  }

  inline void find_min() {
    par_for (U i = 0; i < n; i++) {
      U min_i = 0;
      T min_w = inf;
      for (auto j = edges[i].begin(); j != edges[i].end(); j++) {
        U v = j->v;
        if (rep[i] == rep[v]) continue;
        T w = j->w;
        if (w < min_w) {
          min_i = std::distance(edges[i].begin(), j);
          min_w = w;
        }
        if (w < end_ind[rep[v]].w) {
          pwrite(end_ind[rep[v]], *j);
        }
      }
      if (min_w < inf) {
        pwrite(end_ind[rep[i]], edges[i][min_i]);
      }
    }
  }

//  inline U relabel() {
//    U count_relabel = 0;
//    for (U i = 0; i < n; i++) {
//      if (end_ind[i].w == inf) continue;
//      U this_end = rep[i];
//      U that_end = rep[end_ind[this_end].v];
//      U third_end = rep[end_ind[that_end].v];
//      if (third_end != this_end || this_end > that_end) {
//        rep[this_end] = that_end;
//        edge_set.push_back(end_ind[this_end]);
//        count_relabel++;
//      }
//    }
//    return count_relabel;
//  }

  inline U relabel3() {
    U count_relabel = 0;
    for (U i = 0; i < n; i++) {
      if (end_ind[i].w == inf) continue;
      U this_end = rep[i];
      U that_end, third_end;
      if (this_end == rep[end_ind[this_end].u]) that_end = rep[end_ind[this_end].v];
      else                                      that_end = rep[end_ind[this_end].u];
      if (that_end == rep[end_ind[that_end].u]) third_end = rep[end_ind[that_end].v];
      else                                      third_end = rep[end_ind[that_end].u];
      if (third_end != this_end || this_end > that_end) {
        rep[this_end] = that_end;
        edge_set.push_back(end_ind[this_end]);
        count_relabel++;
      }
    }
    return count_relabel;
  }

//  inline U relabel2() {
//    U count_relabel = 0;
//    auto new_end_ind = std::vector<edge_t>(n, edge_t(0,0,inf));
//    for (U i = 0; i < n; i++) {
//      if (end_ind[i].w == inf) continue;
//      U u = end_ind[i].u;
//      new_end_ind[u] = end_ind[i];
//    }
//    end_ind = new_end_ind;
//    for (U i = 0; i < n; i++) {
//      if (end_ind[i].w == inf) continue;
//      U this_end_rep = rep[i];
//      U that_end = end_ind[i].v;
//      U that_end_rep = rep[that_end];
//      U third_end_rep = rep[end_ind[that_end].v];
//      if (third_end_rep != this_end_rep || this_end_rep > that_end_rep) {
//        rep[this_end_rep] = that_end_rep;
//        edge_set.push_back(end_ind[i]);
//        count_relabel++;
//      }
//    }
//    return count_relabel;
//  }

//  inline void shrink() {
//    for (U i = 0; i < n; i++) {
//      U root = i;
//      while (root != rep[root])
//        root = rep[root];
//      U now = i;
//      U tmp;
//      while (now != root) {
//        tmp = rep[now];
//        rep[now] = root;
//        now = tmp;
//      }
//    }
//  }
  
  inline void pointer_jump() {
    // pointer jump: seems slightly slower, even in parallel
    par_for (U i = 0; i < n; i++) {
      while (rep[i] != rep[rep[i]])
        rep[i] = rep[rep[i]];
//      label[i] = rep[this_label];
    }
  }

  inline void pwrite(edge_t & ptr, edge_t const & new_val) {
    edge_t old_val;
    do {
      old_val = ptr;
    } while (new_val.w < old_val.w and !reinterpret_cast<std::atomic<edge_t>&>(ptr).compare_exchange_strong(old_val, new_val));
  }

//  inline void compact() {
//    par_for (U i = 0; i < n; i++) {
//      U ind = 0;
//      for (U j = 0; j < length[i]; j++) {
//        if (rep[i] != rep[edges[i][j].v]) {
////          if (ind != j)
//          edges[i][ind] = edges[i][j];
//          ind++;
//        }
//      }
//      length[i] = ind;
//    }
//  }

  int output_check() {
    std::vector<U> rep_check(n);
    std::iota(rep_check.begin(), rep_check.end(), 0);
    for (auto const &edge : edge_set) {
      U u = edge.u;
      U v = edge.v;
//      std::cout << u << "," << v << "\n";
      auto ru = u;
      while (ru != rep_check[ru]) {
        ru = rep_check[ru];
      }

      auto rv = v;
      while (rv != rep_check[rv])
        rv = rep_check[rv];

      if (ru == rv)
        return -1;

      U now = u;
      U tmp;
      while (now != ru) {
        tmp = rep_check[now];
        rep_check[now] = ru;
        now = tmp;
      }

      now = v;
      while (now != rv) {
        tmp = rep_check[now];
        rep_check[now] = rv;
        now = tmp;
      }

      rep_check[ru] = rv;
    }
    return 0;
  }

  void run() {
    auto start = now_time();
    reset_clock();
    while (true) {
//      std::cout << vertex_left << std::endl;
      find_min();
      profile("1.find_min");

      U count_relabel = relabel3();
      profile("2.relabel");
      if (vertex_left <= count_relabel + 1 or count_relabel == 0) break;
      vertex_left = vertex_left - count_relabel;

      pointer_jump();
      profile("3.shrink");

//      compact();
//      profile("4.compact");

      std::fill(end_ind.begin(), end_ind.end(), edge_t(0,0,inf));
      profile("5.fill");
    }

    sort(edge_set.begin(), edge_set.end(),
         [](edge_t &e1, edge_t &e2) { return e1.w < e2.w; });
    profile("sort");

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(now_time() - start).count() /
                     1000000.
              << "\n";
    std::cout << edge_set.size() << std::endl;
    // sort edges


  }
};
} // namespace hdbscan
#endif
