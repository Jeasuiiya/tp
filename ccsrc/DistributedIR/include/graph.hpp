#ifndef _HDU_GRAPH_GRAPH_H
#define _HDU_GRAPH_GRAPH_H

#include <graph/node.hpp>
#include <numeric>
#include <utility>
#include <vector>

#include "util.h"
namespace hdu {

template<typename T>
class Graph {
 private:
  std::vector<Node<T>> nodes;
  std::map<std::string, Node<T>&> node_map;

 public:
  Graph() {}
  virtual ~Graph() {}
  GEN_ACCESSOR_IN_DEC(std::vector<Node<T>>, nodes)
  void add_node(Node<T> node) {
    node_map.insert(std::pair<std::string, Node<T>&>(node.get_name(), node));
    nodes.push_back(node);
  }
  void add_node(int at, Node<T> node) {
    node_map.insert(std::pair<std::string, Node<T>&>(node.get_name(), node));
    nodes.insert(nodes.begin() + at, node);
  }
  // Node* add_node() {
  //   Node node;
  //   // node_map.insert(std::pair<std::string, Node&>(node.get_name(), node));
  //   nodes.push_back(node);
  //   return &node;
  // }
  Node<T>& get_node(int at) { return nodes.at(at); }
  Node<T>& get_node(std::string name) { return node_map.find(name)->second; }
  std::string to_string() {
    return std::accumulate(
        nodes.begin(), nodes.end(), std::string(),
        [](std::string& s, Node<T>& p) { return s + "\n" + p.to_string(); });
  }
};
}  // namespace hdu

#endif /* ifndef _GRAPH_GRAPH_H \
        */