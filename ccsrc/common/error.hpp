#ifndef _HDU_ERROR_HPP
#define _HDU_ERROR_HPP
#include <string>
enum class Kind { Unknown, Invalid, Unimplemented, Internal };
struct Error {
  Error(Kind kind, std::string text) {
    this->kind = kind;
    this->text = text;
  }
  Kind kind;
  std::string text;
};
#endif /* ifndef _HDU_ERROR_HPP \
 */