#ifndef DBSCAN_SINGLELINKAGETREE_H_
#define DBSCAN_SINGLELINKAGETREE_H_

#include <vector>
#include <unordered_map>
#include <iostream>
#include <tuple>
#include <cstdint>
#include <queue>
#include "Boruvka.hpp"

namespace hdbscan{
template<typename T, typename U>
  class CondensedTree;

template<typename T, typename U>
  class SingleLinkageTree {
    friend CondensedTree<T, U>;
   public:
    SingleLinkageTree(const std::vector<edge_t> &mst, int minimum_cluster_size):
    // first node that has > 0 distance: node_nums_mst_
      nodes_(2 * mst.size() + 1), minimum_cluster_size_(minimum_cluster_size) {

      total_nums_ = mst.size() + 1;
      node_nums_mst_ = mst.size() + 1;
      for (auto& edge : mst) {        
        auto x = edge.u;
        auto y = edge.v;
        combine(x, y, total_nums_);
        nodes_[total_nums_].distance = edge.w;
        ++total_nums_;
      }
    };

    void get_leaves(const U node_id, std::vector<U> &result) const {      
      if (nodes_[node_id].left != -1) {
        get_leaves(nodes_[node_id].left, result);
      }
      if (nodes_[node_id].right != -1) {
        get_leaves(nodes_[node_id].right, result);
      }
      if (nodes_[node_id].left == -1 && nodes_[node_id].right == -1) {
        result.emplace_back(node_id);        
      }
    }
    
    void print() {
      std::cout << "Representatives: " << std::endl;
      for (decltype(nodes_.size()) i = 0; i < nodes_.size(); ++i) {
        std::cout << i << "->" << nodes_[i].rep << std::endl;
      }

      std::cout << std::endl << "Distance: " << std::endl;
      for (decltype(nodes_.size()) i = 0; i < nodes_.size() - node_nums_mst_; ++i) {
        std::cout << i + node_nums_mst_ << ": " << nodes_[i + node_nums_mst_].distance << std::endl;
      }
    };
    
   public:
    struct Node {
      T distance = 0;
      // TODO: change to pointers maybe?
      U left = -1;
      U right = -1;

      int rep = -1;
      int size = 1;
      
    };
    std::vector<Node> nodes_;
    int total_nums_ = 0;
    int node_nums_mst_ = 0;    
    int minimum_cluster_size_ = 0;
    int cluster_num_ = 0;

    U find(U x, int id) {
      U rep = x;
      if (nodes_[x].rep >= 0)
        rep = find(nodes_[x].rep, id);
      nodes_[x].rep = id;
      return rep;
      /*while (nodes_[x].rep >= 0 || nodes_[x].parent >= 0) {
        U rep = 0;
        if (nodes_[x].rep >= 0)
          rep = find(nodes_[x].rep, id);
        else
          rep = find(nodes_[x].parent, id);
        nodes_[x].rep = id;
        return rep;
      }*/
      return x;
    };

    void combine(U x, U y, int id) {
      U rx = find(x, id);
      U ry = find(y, id);
      nodes_[rx].rep = id;
      nodes_[ry].rep = id;
      //nodes_[rx].parent = id;
      //nodes_[ry].parent = id;
      nodes_[id].size = nodes_[rx].size + nodes_[ry].size;
      if (nodes_[id].size >= minimum_cluster_size_) {
        if ((nodes_[rx].size >= minimum_cluster_size_ && nodes_[ry].size >= minimum_cluster_size_) || 
        (nodes_[rx].size < minimum_cluster_size_ && nodes_[ry].size < minimum_cluster_size_)) {
          ++cluster_num_;
        }
      }
      nodes_[id].left = rx;
      nodes_[id].right = ry;
    };

  };  
}

#endif