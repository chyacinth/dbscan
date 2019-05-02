#ifndef DISTRIBUTED_BORUVKA_H_
#define DISTRIBUTED_BORUVKA_H_
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

#include "DistData.hpp"

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

void my_prod(void *in_, void *inout_, int *len, MPI_Datatype *dptr) {
  edge_t *in = static_cast<edge_t*>(in_);
  edge_t *inout = static_cast<edge_t*>(inout_);
  for (int i = 0; i < *len; ++i) {
    if (in->w < inout->w) {
      inout->u = in->u;
      inout->v = in->v;
      inout->w = in->w;
    }
    in++;
    inout++;
  }
}

template <typename T, typename U> class Distributed_Boruvka : public mpi::MPIObject{

  constexpr static auto now_time = std::chrono::high_resolution_clock::now;
  using edge_p = std::pair<T, size_t>;

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
  const U m, n;
  U local_n;
  U vertex_left;
  std::vector<edge_p> edges2;
  DistData<STAR, CBLK, pair<T, size_t>>& edges;

  std::vector<edge_t> edge_set;
  std::vector<uint32_t > rep;
  std::vector<edge_t> end_ind;
  MPI_Datatype etype;
  MPI_Op my_op;

  std::map<std::string, double> profiler;

  inline T get_weight(std::pair<T, T> const & a, std::pair<T, T> const & b) {
    return std::pow(std::pow(a.first - b.first, 2.) + std::pow(a.second - b.second, 2.), 0.5);
  };



  Distributed_Boruvka(DistData<STAR, CBLK, pair<T, size_t>>& edges_, mpi::Comm comm_) : mpi::MPIObject(comm_),
      m(edges_.row()), n(edges_.col()), vertex_left(edges_.col()), edges(edges_) {
    size_t edge_n = n % GetCommSize();
    local_n = ( n - edge_n ) / GetCommSize();
    edges2 = std::vector<edge_p>(size_t(local_n) * m);
    random_device rd;
    std::mt19937 gen(56419847);
    std::uniform_real_distribution<T> pos_gen(0.0, 1000.0);
    std::uniform_int_distribution<U> id_gen(0, n-1);

    /*
    // random initialization
    std::vector<std::pair<T, T>> points(n);
    for (size_t i = 0; i < n; i++) {
      points[i] = {pos_gen(gen), pos_gen(gen)};
    }
    for (size_t i = 0; i < local_n; i++) {
      std::unordered_set<U> gen_set;
      for (size_t j = 0; j < m; j++) {
        U g;
        do {
          g = id_gen(gen);
        } while (g == GetCommRank() + i * GetCommSize() or gen_set.find(g) != gen_set.end());
        gen_set.insert(g);
        edges2[i * m + j] = edge_p(get_weight(points[GetCommRank() + i * GetCommSize()], points[g]), g);
        //edges2[j * local_n + i] = edge_p(get_weight(points[GetCommRank() + i * GetCommSize()], points[g]), g);
      }
    }
    edges = DistData<STAR, CBLK, pair<T, size_t>>(m, n, edges2, GetComm());*/

    /*if (GetCommRank() == 1)
    for ( int i = 0; i < std::min( m, (U)100 ); i ++ ) {
      for (int j = 0; j < std::min(local_n, (U)100); ++j)
        printf( "[%5.2lf,%5lu] ",edges( i, GetCommRank() + j * GetCommSize() ).first, edges( i, GetCommRank() + j * GetCommSize() ).second);
      printf("\n");
      printf("OK\n");
    }*/

    end_ind = std::vector<edge_t>(n, edge_t(0,0,inf));
    rep = std::vector<U>(n);
    std::iota(rep.begin(), rep.end(), 0);

    // init MPI

    int block_length[3] = {1, 1, 1};
    MPI_Aint displacements[3] = {offsetof(edge_t, u), offsetof(edge_t, v), offsetof(edge_t, w)};
    MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT32_T, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_length, displacements, types, &etype);
    MPI_Type_commit(&etype);
    MPI_Op_create(my_prod, 0, &my_op);
  }

  inline void find_min() {
    par_for (size_t k = 0; k < local_n; k++) {
      size_t i = GetCommRank() + k * GetCommSize();
      U min_i = 0;
      T min_w = inf;
      for (size_t j = 0; j < m; j++) {
        //U v2 = edges2[i * m + j].second;
        U v = edges(j, i).second;
        if (rep[i] == rep[v]) continue;
        //T w2 = edges2[i * m + j].first;
        T w = edges(j, i).first;
        if (w < min_w) {
          min_i = j;
          min_w = w;
        }
        if (w < end_ind[rep[v]].w) {
          pwrite(end_ind[rep[v]], edge_t(static_cast<U>(i), v, w));
        }
      }
      if (min_w < inf) {
        //auto const & e = edges2[i * m + min_i];
        auto const & e = edges(min_i, i);
        pwrite(end_ind[rep[i]], edge_t(U(i), static_cast<U>(e.second), e.first));
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
    par_for (size_t i = 0; i < n; i++) {
      int cnt = 0;
      while (rep[i] != rep[rep[i]]) {
        ++cnt;
        rep[i] = rep[rep[i]];
      }

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
    int inspect = GetCommSize() + 1;
    auto start = now_time();
    reset_clock();
    while (true) {
      find_min();
      profile("1.find_min");

      if (GetCommRank() == inspect) {
        std::cout << "after find_min " << GetCommRank() << std::endl;
      }

      edge_t *ei = end_ind.data();

      if (GetCommRank() == 0) {
        MPI_Reduce(MPI_IN_PLACE, ei, static_cast<int>(n), etype, my_op, 0, GetComm());
      } else {
        MPI_Reduce(ei, ei, static_cast<int>(n), etype, my_op, 0, GetComm());
      }

      if (GetCommRank() == inspect) {
        std::cout << "after reduce " << GetCommRank() << std::endl;
      }

      bool end_loop = false;
      if (GetCommRank() == 0) {
        U count_relabel = relabel3();
        profile("2.relabel");

        if (GetCommRank() == inspect) {
          std::cout << "after relabel " << GetCommRank() << std::endl;
          std::cout << count_relabel << std::endl;
        }

        if (vertex_left <= count_relabel + 1 or count_relabel == 0) {
          end_loop = true;
        } else {
          vertex_left = vertex_left - count_relabel;

          if (GetCommRank() == inspect)
            std::cout << "before shrink " << GetCommRank() << std::endl;

          pointer_jump();
          profile("3.shrink");

          if (GetCommRank() == inspect) {
            std::cout << "after shrink " << GetCommRank() << std::endl;
            for (int i = 0; i < n; ++i) {
              std::cout << rep[i] << std::endl;
            }
          }
        }
      }

//      compact();
//      profile("4.compact");

      uint32_t *r = rep.data();
      // TODO: remove barrier?
      // MPI_Barrier(GetComm());
      uint32_t tmp = 0;
      if (end_loop) {
        tmp = r[0];
        r[0] = static_cast<uint32_t >(n);
        //MPI_Bcast(r, n, MPI_UINT32_T, 0, GetComm());
      }

      MPI_Bcast(r, n, MPI_UINT32_T, 0, GetComm());

      if (GetCommRank() == inspect) {
        std::cout << "after Bcast " << GetCommRank() << std::endl;
        std::cout << "edge_set size: " << edge_set.size() << std::endl;
      }

      if (end_loop || r[0] == static_cast<uint32_t>(n)) {
        r[0] = tmp;
        break;
      }

      std::fill(end_ind.begin(), end_ind.end(), edge_t(0,0,inf));
      profile("5.fill");
      if (GetCommRank() == inspect)
        std::cout << "after fill " << GetCommRank() << std::endl;
    }
    if (GetCommRank() == 0) {
      sort(edge_set.begin(), edge_set.end(),
           [](edge_t &e1, edge_t &e2) { return e1.w < e2.w; });
      profile("sort");

      std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
          now_time() - start).count() /
          1000000.
                << "\n";
      std::cout << edge_set.size() << std::endl;
      std::cout << output_check() << std::endl;
    }
    // sort edges


  }
};
} // namespace hdbscan
#endif
