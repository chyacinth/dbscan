#ifndef DBSCAN_CONDENSEDTREE_H_
#define DBSCAN_CONDENSEDTREE_H_

#include "SingleLinkageTree.hpp"

namespace hdbscan {
  template<typename T, typename U>
  class CondensedTree
  {  
   public:
    explicit CondensedTree(const SingleLinkageTree<T, U> &slt)
      : clusters_(slt.cluster_num_), point_nums_(slt.node_nums_mst_),
      minimum_cluster_size_(slt.minimum_cluster_size_) {
      int root = slt.nodes_.size() - 1;
      //clusters_.emplace_back();
      clusters_[0].lambda_birth = 1 / slt.nodes_[root].distance;
      clusters_[0].remaining_nodes_num = slt.nodes_[root].size;
#pragma omp parallel
#pragma omp single
      {
        //std::cout << "start build" << endl;
        //std::cout << "cluster_num_: " << slt.cluster_num_ << std::endl;
        build(slt, root, cluster_num_++);
        //std::cout << "end build" << endl;
        //std::cout << "start extract" << endl;
        extract_clusters();
        //std::cout << "end extract" << endl;
      }
    }
    void build(const SingleLinkageTree<T, U> &slt, const U root, const U cluster_id) {
      const auto& node = slt.nodes_[root];
      bool keep_left = false;
      bool keep_right = false;      
      const int left_size = slt.nodes_[node.left].size;
      const int right_size = slt.nodes_[node.right].size;
      //std::cout << "root_id: " << root << std::endl;
      //std::cout << "node.left: " << node.left << std::endl;      
      //std::cout << "node.right: " << node.right << std::endl;      
      if (cluster_id > clusters_.size() - 1) {
        std::cout << "cluster_id: " << cluster_id << std::endl;
        std::cout << "clusters_.size(): " << clusters_.size() << std::endl;
        std::cout << "wtf" << std::endl;
      }
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

//#pragma omp task
          build(slt, node.left, left_cluster);
//#pragma omp task
          build(slt, node.right, right_cluster);
//#pragma omp taskwait
      }
    }

    void extract_clusters() {
      selected_ = std::vector<char>(cluster_num_);
      //std::cout << "start 1" << std::endl;
      update_stability(0);
      //std::cout << "start 2" << std::endl;
      selected_cnt_ = select_helper(0);
      //std::cout << "start 3" << std::endl;
      collect_helper(0);
      //std::cout << "start 4" << std::endl;
//#pragma omp parallel for
      for (U i = 0; i < cluster_num_; ++i) {
        if (selected_[i]) {
          int left_result_size = (clusters_[i].left == -1)? 0 : clusters_[clusters_[i].left].result_size;
          int ori_size = clusters_[i].fall_out_nodes.size();
          clusters_[i].fall_out_nodes.resize(clusters_[i].result_size + 1);
          //clusters_[i].fall_out_nodes.resize(clusters_[i].result_size * 10);
#pragma omp task
          collect_points2(clusters_[i].left, clusters_[i].fall_out_nodes.data() + ori_size);
#pragma omp task
          collect_points2(clusters_[i].right, clusters_[i].fall_out_nodes.data() + ori_size + left_result_size);
#pragma omp taskwait
        }        
      }
      //std::cout << "start 5" << std::endl;

    }

    T update_stability(U root_id) {
      if (root_id == -1) return 0;
      auto& root_node = clusters_[root_id];
      // leaf node, just return
      if (root_node.left == -1 && root_node.right == -1) {
        selected_[root_id] = true;        
        return root_node.stability;
      }
      // get left and right stability
      T left_stability = 0;
      T right_stability = 0;
#pragma omp task shared(left_stability)
        left_stability = update_stability(root_node.left);

#pragma omp task shared(right_stability)
      right_stability = update_stability(root_node.right);

#pragma omp taskwait
      if (left_stability + right_stability >= root_node.stability) {
        root_node.stability = left_stability + right_stability;
      } else if (root_id != 0) {
        selected_[root_node.left] = false;
        selected_[root_node.right] = false;
        selected_[root_id] = true;
      }
      //std::cout << root_id << "' s stability is: " << root_node.stability << std::endl;
      return root_node.stability;
    }

    int collect_helper(U root) {
      if (root == -1) return 0;
      int lsize = 0;
      int rsize = 0;
#pragma omp task shared(lsize)
      lsize = collect_helper(clusters_[root].left);
#pragma omp task shared(rsize)
      rsize = collect_helper(clusters_[root].right);
#pragma omp taskwait
      clusters_[root].result_size = clusters_[root].fall_out_nodes.size() + lsize + rsize;
      return clusters_[root].result_size;
    }

    void collect_points2(U root, U* result) {
      if (root == -1) return;
      for (auto i : clusters_[root].fall_out_nodes) {
        *result = i;
        ++result;
      }

      int left_result_size = (clusters_[root].left == -1)? 0 : clusters_[clusters_[root].left].result_size;
#pragma omp task
      collect_points2(clusters_[root].left, result);
#pragma omp task
      collect_points2(clusters_[root].right, result + left_result_size);
#pragma omp taskwait
    }

    void collect_points(U root, std::vector<U>& result) {
      if (root == -1) return;
      for (auto i : clusters_[root].fall_out_nodes) {
        result.emplace_back(i);
      }
      collect_points(clusters_[root].left, result);
      collect_points(clusters_[root].right, result);
    }

    void print(bool verbose = false, bool store = false) {
      std::cout << "------------Clustering Result------------" << std::endl;
      std::cout << "Total cluster number: " << cluster_num_ << std::endl;
      if (store) {
        point_cluster_ = std::vector<int>(point_nums_ + 1, -1);
      }

      print_helper(0, verbose, store);

      std::cout << "Selected cluster number: " << selected_cnt_ << std::endl;
    }

    std::vector<int> get_point_cluster() {
      return point_cluster_;
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
      int result_size = 0;
    };
    std::vector<Node> clusters_;
    std::vector<char> selected_;
    std::vector<int> point_cluster_;
    int cluster_num_ = 0;
    int point_nums_ = 0;
    int minimum_cluster_size_ = 0;
    int selected_cnt_ = 0;


    int select_helper(U cluster_id) {
      if (cluster_id == -1) {
        return 0;
      }
      if (selected_[cluster_id]) {
        return 1;
      } else {
        int lsize = 0;
#pragma omp task shared(lsize)
        lsize = select_helper(clusters_[cluster_id].left);
        int rsize = 0;
#pragma omp task shared(rsize)
        rsize = select_helper(clusters_[cluster_id].right);
#pragma omp taskwait
        return lsize + rsize;
      }
    }


    void print_helper(U cluster_id, bool verbose, bool store) {
      if (cluster_id == -1) {
        return;
      }
      if (selected_[cluster_id]) {
        if (verbose) {
          std::cout << "Cluster id: " << cluster_id << std::endl;
          for (auto node : clusters_[cluster_id].fall_out_nodes) {
            std::cout << node << " ";
          }
          std::cout << std::endl;
        }
        if (store) {
          for (auto node : clusters_[cluster_id].fall_out_nodes) {
            point_cluster_[node] = cluster_id;
          }
        }
        return;
      } else {
        print_helper(clusters_[cluster_id].left, verbose, store);
        print_helper(clusters_[cluster_id].right, verbose, store);
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