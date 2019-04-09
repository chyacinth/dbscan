#ifndef DBSCAN_CONDENSEDTREE_H_
#define DBSCAN_CONDENSEDTREE_H_

#include "SingleLinkageTree.hpp"

namespace hdbscan {
  template<typename T, typename U>
  class CondensedTree
  {  
   public:
    CondensedTree(const SingleLinkageTree<T, U> &slt, int minimum_cluster_size)
      : minimum_cluster_size_(minimum_cluster_size) {
      int root = slt.nodes_.size() - 1;
      clusters_.emplace_back();      
      clusters_[0].lambda_birth = 1 / slt.nodes_[root].distance;
      build(slt, root, cluster_num_++);
      extract_clusters();
    }
    void build(const SingleLinkageTree<T, U> &slt, U root, U cluster_id) {
      auto& node = slt.nodes_[root];
      bool keep_left = false;
      bool keep_right = false;

      if (node.left != -1 && slt.nodes_[node.left].size >= minimum_cluster_size_) {
        keep_left = true;
      }
      if (node.right != -1 && slt.nodes_[node.right].size >= minimum_cluster_size_) {
        keep_right = true;
      }

      if (!keep_left && node.left != -1) {
        slt.get_leaves(node.left, node.distance, clusters_[cluster_id].fall_out_nodes_);
      }
      if (!keep_right && node.right != -1) {
        slt.get_leaves(node.right, node.distance, clusters_[cluster_id].fall_out_nodes_);
      }

      if (keep_left && !keep_right) {        
        build(slt, node.left, cluster_id);
      }

      if (keep_right && !keep_left) {        
        build(slt, node.right, cluster_id);
      }

      if (keep_left && keep_right) {
        U left_cluster = cluster_num_++;
        clusters_.emplace_back();
        clusters_[left_cluster].lambda_birth = 1 / node.distance;
        U right_cluster = cluster_num_++;
        clusters_.emplace_back();
        clusters_[right_cluster].lambda_birth = 1 / node.distance;
        clusters_[cluster_id].left = left_cluster;
        clusters_[cluster_id].right = right_cluster;

        build(slt, node.left, left_cluster);
        build(slt, node.right, right_cluster);

        for (auto node_lam : clusters_[left_cluster].fall_out_nodes_) {
          clusters_[cluster_id].fall_out_nodes_.emplace_back(node_lam.first, 1 / node.distance);
        }
        for (auto node_lam : clusters_[right_cluster].fall_out_nodes_) {
          clusters_[cluster_id].fall_out_nodes_.emplace_back(node_lam.first, 1 / node.distance);
        }
      }

      /*if (!keep_left && !keep_right) {
        slt.get_leaves(root, node.distance, clusters_[cluster_id].fall_out_nodes_);
      }*/

    }

    void extract_clusters() {
      selected_ = std::vector<char>(cluster_num_);
      for (U i = 0; i < cluster_num_; ++i)
        clusters_[i].stability = calc_stability(i);
      update_stability(0);
    }

    T update_stability(U root_id) {
      auto& root_node = clusters_[root_id];
      // leaf node, just return
      if (root_node.left == -1 && root_node.right == -1) {
        selected_[root_id] = true;        
        return root_node.stability;
      }
      // get left and right stability
      T left_stability = 0;
      if (root_node.left != -1) {
        left_stability = update_stability(root_node.left);
      }
      T right_stability = 0;
      if (root_node.right != -1) {
        right_stability = update_stability(root_node.right);
      }
      if (left_stability + right_stability > root_node.stability) {        
        root_node.stability = left_stability + right_stability;
      } else {
        selected_[root_node.left] = false;
        selected_[root_node.right] = false;
        selected_[root_id] = true;
      }
      return root_node.stability;
    }
    
    void print() {
      std::cout << "------------Clustering Result------------" << std::endl;
      for (int i = 0; i < cluster_num_; ++i) {
        if (selected_[i]) {
          std::cout << "Cluster id: " << i << std::endl;        
          for (auto node_lambda : clusters_[i].fall_out_nodes_) {
            std::cout << node_lambda.first << " ";
          }
          std::cout << std::endl;
        }
      }
    }
   private:
    struct Node {
      U left = -1;
      U right = -1;
      // [node_id, lambda]
      std::vector<std::pair<U, T>> fall_out_nodes_;
      T stability = 0;
      T lambda_birth = 0;      
    };
    std::vector<Node> clusters_;
    std::vector<char> selected_;
    int cluster_num_ = 0;
    int minimum_cluster_size_ = 0;

    T calc_stability(U cluster_id) {
      T result = 0;
      T lambda_birth = clusters_[cluster_id].lambda_birth;
      for (auto node_lambda : clusters_[cluster_id].fall_out_nodes_) {
        result += node_lambda.second - lambda_birth;
      }
      return result;
    }
  };  
  
}
#endif