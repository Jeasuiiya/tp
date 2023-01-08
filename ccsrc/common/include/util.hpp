#ifndef _FRAMEWORK_UTIL_H
#define _FRAMEWORK_UTIL_H

#define ALL(...) __VA_ARGS__
#ifndef GEN_SETTER
#define GEN_SETTER(cName, fType, fName) \
    inline void cName::set##_##fName(fType fName) { this->fName = fName; }
#endif /* ifndef GEN_SETTER(cName, fName, fType) */

#ifndef GEN_SETTER_IN_DEC
#define GEN_SETTER_IN_DEC(fType, fName) \
    inline void set##_##fName(fType fName) { this->fName = std::move(fName); }
#endif /* ifndef GEN_SETTER(cName, fName, fType) */

#ifndef GEN_GETTER_IN_DEC
#define GEN_GETTER_IN_DEC(fType, fName) \
    inline fType& get##_##fName() { return fName; }

#endif /* ifndef GEN_GETTER_IN_DEC */

#ifndef GEN_ACCESSOR_IN_DEC
#define GEN_ACCESSOR_IN_DEC(fType, fName) \
    GEN_SETTER_IN_DEC(ALL(fType), fName)  \
    GEN_GETTER_IN_DEC(ALL(fType), fName)
#endif /* ifndef GEN_ACCESSOR_IN_DEC(fType, fName) \
        */

#ifndef GEN_PROXY_SETTER
#define GEN_PROXY_SETTER(fType, Proxy, fName) \
    inline void set##_##fName(fType fName) {  \
        this->Proxy->set##_##fName(fName);    \
    }
#endif /* ifndef GEN_PROXY_SETTER(cName, fName, fType) */

#ifndef GEN_PROXY_GETTER
#define GEN_PROXY_GETTER(fType, Proxy, fName) \
    inline fType& get##_##fName() { return this->Proxy->get##_##fName(); }
#endif /* ifndef GEN_PROXY_GETTER */

#ifndef GEN_PROXY_ACCESSOR
#define GEN_PROXY_ACCESSOR(fType, Proxy, fName) \
    GEN_PROXY_SETTER(ALL(fType), Proxy, fName)  \
    GEN_PROXY_GETTER(ALL(fType), Proxy, fName)
#endif /* ifndef GEN_PROXY_ACCESSOR(fType, fName) \
        */

#define HDU_STATUS_MACROS_CONCAT_NAME(x, y) HDU_STATUS_MACROS_CONCAT_IMPL(x, y)
#define HDU_STATUS_MACROS_CONCAT_IMPL(x, y) x##y
#define HDU_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
    auto statusor = (rexpr);                            \
    if (statusor.isErr()) {                             \
        return Err(statusor.unwrapErr());               \
    }                                                   \
    lhs = std::move(statusor.unwrap())

#define HDU_ASSIGN_OR_RETURN(lhs, rexpr)                                   \
    HDU_ASSIGN_OR_RETURN_IMPL(                                             \
        HDU_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
        rexpr)

#define HDU_RETURN_IF_ERROR(...)        \
    do {                                \
        auto _r = (__VA_ARGS__);        \
        if (_r.isErr()) {               \
            return Err(_r.unwrapErr()); \
        }                               \
    } while (0)

#endif /* ifndef _HDU_UTIL_H */