#ifndef _HDU_GRAPH_NODE_H
#define _HDU_GRAPH_NODE_H

#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "util.h"
namespace hdu {

class NodeBase {
 private:
  std::string name;
  std::string op;
  std::vector<std::string> inputs;
  std::string device;
  std::map<std::string, std::string> attrs;
  // T data;
 public:
  NodeBase() {}
  virtual ~NodeBase() {}
  GEN_ACCESSOR_IN_DEC(std::string, name)
  GEN_ACCESSOR_IN_DEC(std::string, op)
  GEN_ACCESSOR_IN_DEC(std::string, device)
  GEN_ACCESSOR_IN_DEC(std::vector<std::string>, inputs)
  GEN_ACCESSOR_IN_DEC(TYPE(std::map<std::string, std::string>), attrs)
  // GEN_ACCESSOR_IN_DEC(T, data)
  void add_input(std::string input) { inputs.push_back(input); }
  std::string to_string() {
    std::stringstream ss;
    ss << "name:" << name << std::endl;
    ss << "op:" << op << std::endl;
    ss << "inputs: "
       << std::accumulate(inputs.begin(), inputs.end(), std::string(),
                          [](const std::string& s, const std::string& p) {
                            return s + (s.empty() ? std::string() : ", ") + p;
                          })
       << std::endl;
    ss << "device: " << device << std::endl;
    ss << "attrs: "
       << std::accumulate(
              attrs.begin(), attrs.end(), std::string(),
              [](const std::string& s,
                 const std::pair<const std::string, std::string>& p) {
                return s + (s.empty() ? std::string() : "\n") + p.first + ": " +
                       p.second;
              })
       << std::endl;
    return ss.str();
  }
};
template <typename T>
class Node : public NodeBase {
  using NodeBase::NodeBase;

 public:
  T data;
  GEN_ACCESSOR_IN_DEC(T, data)
};
template <>
class Node<void> : public NodeBase {
  using NodeBase::NodeBase;
};
// GEN_SETTER(Node, std::string, name)
}  // namespace hdu

#endif /* ifndef _GRAPH_NODE_H */