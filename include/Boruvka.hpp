#ifndef BORUVKA_H_
#define BORUVKA_H_
//#define NDEBUG

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

struct edge_t {
  uint32_t u;
  uint32_t v;
  double w;
public:
  edge_t() = default;
  edge_t(uint32_t u_, uint32_t v_, double w_) : u(u_), v(v_), w(w_) {};
};

template <typename T, typename U> class Boruvka {

  constexpr static auto now_time = std::chrono::high_resolution_clock::now;
  using edge_p = std::pair<T, size_t>;

  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
  inline void reset_clock() { timestamp = now_time(); }
  inline void profile(std::string name, bool p=false) {
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(now_time() - timestamp).count() /
        1000000.;
    if (p) std::cout << vertex_left << "\t" << name << "\t" << time << std::endl;
    profiler[name] += time;
    timestamp = now_time();
  }

  inline T get_weight(std::pair<T, T> const & a, std::pair<T, T> const & b) {
    return std::pow(std::pow(a.first - b.first, 2.) + std::pow(a.second - b.second, 2.), 0.5);
  };

  void init() {
    end_ind = std::vector<edge_t>(n, edge_t(0,0,inf));
    rep = std::vector<U>(n);
    std::iota(rep.begin(), rep.end(), 0);
  }

  void rand_init_edges() {
    edges = std::vector<edge_p>(size_t(n) * m);
    // random_device rd;
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> pos_gen(0.0, 1000.0);
    std::uniform_int_distribution<U> id_gen(0, n-1);

    // random initialization
    std::vector<std::pair<T, T>> points(n);
    for (size_t i = 0; i < n; i++) {
      points[i] = {pos_gen(gen), pos_gen(gen)};
    }
    for (size_t i = 0; i < n; i++) {
      std::unordered_set<U> gen_set;
      for (size_t j = 0; j < m; j++) {
        U g;
        do {
          g = id_gen(gen);
        } while (g == i or gen_set.find(g) != gen_set.end());
        edges[i * m + j] = edge_p(get_weight(points[i], points[g]), g);
      }
    }
  }

public:
  T inf = std::numeric_limits<T>::infinity();
  const U n, m;
  U vertex_left;
  std::vector<edge_p> edges;
  std::vector<edge_t> edge_set;
  std::vector<U> rep;
  std::vector<edge_t> end_ind;

  std::map<std::string, double> profiler;

  Boruvka(U n_, U m_) : n(n_), m(m_), vertex_left(n_) {
    rand_init_edges();
    init();
  }

  Boruvka(U n_, U m_, std::vector<edge_p> & E) : n(n_), m(m_), vertex_left(n_) {
    assert(n * m == E.size());
    edges.swap(E);
    init();
  }

  inline void find_min() {
    par_for (size_t i = 0; i < n; i++) {
      U min_i = 0;
      T min_w = inf;
      for (size_t j = 0; j < m; j++) {
        U v = edges[i * m + j].second;
        if (rep[i] == rep[v]) continue;
        T w = edges[i * m + j].first;
        if (w < min_w) {
          min_i = j;
          min_w = w;
        }
        if (w < end_ind[rep[v]].w) {
          pwrite(end_ind[rep[v]], edge_t(U(i), v, w));
        }
      }
      if (min_w < inf) {
        auto const & e = edges[i * m + min_i];
        pwrite(end_ind[rep[i]], edge_t(U(i), e.second, e.first));
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
    for (size_t i = 0; i < n; i++) {
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
    par_for (size_t i = 0; i < n; i++)
      while (rep[i] != rep[rep[i]])
        rep[i] = rep[rep[i]];
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
      if (vertex_left <= count_relabel + 1) break;
      vertex_left = vertex_left - count_relabel;

      pointer_jump();
      profile("3.shrink");
      if (count_relabel == 0) {
        std::unordered_set<U> s;
        s.insert(rep[0]);
        U last = rep[0];
        for (size_t i = 1; i < n; i++) {
          if (s.find(rep[i]) != s.end()) continue;
          edge_set.push_back(edge_t(last, rep[i], inf));
          last = rep[i];
          s.insert(last);
        }
        break;
      }

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
