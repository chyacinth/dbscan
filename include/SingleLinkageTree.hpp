#ifndef DBSCAN_SINGLELINKAGETREE_H_
#define DBSCAN_SINGLELINKAGETREE_H_

#include <vector>
#include <unordered_map>
#include <iostream>
#include <cstdint>

namespace dbscan{
template<typename T, typename U = uint32_t>
  class SingleLinkageTree {
   public:
    SingleLinkageTree(const vector<tuple<U, U, T>> &mst): 
        heights_(2 * mst.size()), parent_(2 * mst.size(), -1), rep_(2 * mst.size(), -1_) {
      total_nums_ = mst.size();
      for (int i = 0; i < mst.size(); ++i) {
        auto edge = &mst[i];
        auto x = std::get<0>(edge);
        auto y = std::get<1>(edge);
        combine(x, y, total_nums);
        distance_[total_nums] = std::get<2>(edge);
        ++total_nums;
      }
    };
    void print() {
      std::cout << "Parents: " << std::endl;
      for (int i = 0; i < parent_.size(); ++i) {
        std::cout << i << "->" << parent_[i] << std::endl;
      }
      std::cout << std::endl << "Distance: " << std::endl;
      for (int i = 0; i < distance_.size(); ++i) {
        std::cout << i << ": " << distance << std::endl;
      }
    };

   private:
    std::vector<T> distance_;
    std::vector<int> parent_;
    std::vector<int> rep_;   
    int total_nums_ = 0;

    U find(U x, int id) {
      while (rep_[x] >= 0) {
        U rep = find(rep_[x]);
        rep_[x] = id;
        return rep;
      }
    };

    void combine(U x, U y, int id) {
      U rx = find(x);
      U ry = find(y);
      rep_[rx] = id;
      rep_[ry] = id;
      parent_[rx] = id;
      parent_[ry] = id;
    };

  };
};

#endif