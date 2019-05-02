#ifndef DBSCAN_CONDENSEDTREE_H_
#define DBSCAN_CONDENSEDTREE_H_

#include "SingleLinkageTree.hpp"

namespace hdbscan {
  template<typename T, typename U>
  class CondensedTree
  {  
   public:
    CondensedTree(const SingleLinkageTree<T, U> &slt)
      : clusters_(slt.cluster_num_), minimum_cluster_size_(slt.minimum_cluster_size_) {      
      int root = slt.nodes_.size() - 1;
      //clusters_.emplace_back();
      clusters_[0].lambda_birth = 1 / slt.nodes_[root].distance;
      clusters_[0].remaining_nodes_num = slt.nodes_[root].size;
      build(slt, root, cluster_num_++);
      extract_clusters();
    }
    void build(const SingleLinkageTree<T, U> &slt, const U root, const U cluster_id) {
      const auto& node = slt.nodes_[root];
      bool keep_left = false;
      bool keep_right = false;      
      const int left_size = slt.nodes_[node.left].size;
      const int right_size = slt.nodes_[node.right].size;

      if (node.left != -1 && left_size >= minimum_cluster_size_) {
        keep_left = true;
      }
      if (node.right != -1 && right_size >= minimum_cluster_size_) {
        keep_right = true;
      }

      int fall_out_num = 0;

      if (!keep_left && node.left != -1) {
        fall_out_num += left_size;
        slt.get_leaves(node.left, clusters_[cluster_id].fall_out_nodes);
      }
      if (!keep_right && node.right != -1) {
        fall_out_num += right_size;
        slt.get_leaves(node.right, clusters_[cluster_id].fall_out_nodes);
      }

      if (keep_left && !keep_right) {
        clusters_[cluster_id].stability += (1 / node.distance - clusters_[cluster_id].lambda_birth) * clusters_[cluster_id].remaining_nodes_num;
        clusters_[cluster_id].remaining_nodes_num -= fall_out_num;
        clusters_[cluster_id].lambda_birth = 1 / node.distance;
        build(slt, node.left, cluster_id);
      }

      if (keep_right && !keep_left) {
        clusters_[cluster_id].stability += (1 / node.distance - clusters_[cluster_id].lambda_birth) * clusters_[cluster_id].remaining_nodes_num;
        clusters_[cluster_id].remaining_nodes_num -= fall_out_num;
        clusters_[cluster_id].lambda_birth = 1 / node.distance;
        build(slt, node.right, cluster_id);
      }

      if (keep_left && keep_right) {
        U left_cluster = cluster_num_++;
        //clusters_.emplace_back();
        clusters_[left_cluster].lambda_birth = 1 / node.distance;
        clusters_[left_cluster].remaining_nodes_num = left_size;
        U right_cluster = cluster_num_++;
        //clusters_.emplace_back();
        clusters_[right_cluster].lambda_birth = 1 / node.distance;
        clusters_[right_cluster].remaining_nodes_num = right_size;

        clusters_[cluster_id].left = left_cluster;
        clusters_[cluster_id].right = right_cluster;

        clusters_[cluster_id].stability += (1 / node.distance - clusters_[cluster_id].lambda_birth) * clusters_[cluster_id].remaining_nodes_num;
        
        clusters_[cluster_id].remaining_nodes_num = 0;
        clusters_[cluster_id].lambda_birth = 1 / node.distance;

        build(slt, node.left, left_cluster);
        build(slt, node.right, right_cluster);

        /*for (auto node_lam : clusters_[left_cluster].fall_out_nodes) {
          clusters_[cluster_id].fall_out_nodes.emplace_back(node_lam.first, 1 / node.distance);
        }
        for (auto node_lam : clusters_[right_cluster].fall_out_nodes) {
          clusters_[cluster_id].fall_out_nodes.emplace_back(node_lam.first, 1 / node.distance);
        }*/
      }

      /*if (!keep_left && !keep_right) {
        slt.get_leaves(root, clusters_[cluster_id].fall_out_nodes);
      }*/

    }

    void extract_clusters() {
      selected_ = std::vector<char>(cluster_num_);

      update_stability(0);
      
      for (U i = 0; i < cluster_num_; ++i) {
        if (selected_[i]) {
          collect_points(clusters_[i].left, clusters_[i].fall_out_nodes);
          collect_points(clusters_[i].right, clusters_[i].fall_out_nodes);
        }        
      }
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
    
    void collect_points(U root, std::vector<U>& result) {
      if (root == -1) return;
      for (auto i : clusters_[root].fall_out_nodes) {
        result.emplace_back(i);
      }
      collect_points(clusters_[root].left, result);
      collect_points(clusters_[root].right, result);
    }

    void print(bool verbose = false) {
      std::cout << "------------Clustering Result------------" << std::endl;
      std::cout << "Total cluster number: " << cluster_num_ << std::endl;

      int selected_cnt = print_helper(0, verbose);

      std::cout << "Selected cluster number: " << selected_cnt << std::endl;
    }

   private:
    struct Node {
      U left = -1;
      U right = -1;
      // [node_id, lambda]
      std::vector<U> fall_out_nodes;
      T stability = 0;
      T lambda_birth = 0;
      int remaining_nodes_num = 0;
    };
    std::vector<Node> clusters_;
    std::vector<char> selected_;
    int cluster_num_ = 0;
    int minimum_cluster_size_ = 0;

    int print_helper(U cluster_id, bool verbose) {
      if (cluster_id == -1) {
        return 0;
      }
      if (selected_[cluster_id]) {
        if (verbose) {
          std::cout << "Cluster id: " << cluster_id << std::endl;
          for (auto node : clusters_[cluster_id].fall_out_nodes) {
            std::cout << node << " ";
          }
          std::cout << std::endl;
        }
        return 1;
      } else {
        int lsize = print_helper(clusters_[cluster_id].left, verbose);
        int rsize = print_helper(clusters_[cluster_id].right, verbose);
        return lsize + rsize;
      }
    }
    T calc_stability(U cluster_id) {
      T result = 0;
      T lambda_birth = clusters_[cluster_id].lambda_birth;
      for (auto node_lambda : clusters_[cluster_id].fall_out_nodes) {
        result += node_lambda.second - lambda_birth;
      }
      return result;
    }
  };  
  
}
#endif