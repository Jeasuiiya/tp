#ifndef FRAMEWORK_UTIL_H
#define FRAMEWORK_UTIL_H

#define ALL(...) __VA_ARGS__

#ifndef DECL_SETTER
#define DECL_SETTER(SetterName, Type, fieldName, MOVE) \
    /* NOLINTNEXTLINE */                               \
    inline void SetterName(Type && (fieldName)) {      \
        if constexpr (MOVE) {                          \
            this->fieldName = std::move(fieldName);    \
        } else {                                       \
            this->fieldName = fieldName;               \
        }                                              \
    }
#endif

#ifndef DECL_SETTER_M
#define DECL_SETTER_M(SetterName, Type, fieldName) \
    /* NOLINTNEXTLINE */                           \
    inline void SetterName(Type&(fieldName)) { this->fieldName = std::move(fieldName); }
#endif

#ifndef DECL_GETTER
#define DECL_GETTER(GetterName, Type, fieldName) \
    inline Type& GetterName() { return fieldName; }
#endif

#ifndef DECL_ACCESSOR
#define DECL_ACCESSOR(GetterName, SetterName, Type, fName, MOVE) \
    DECL_SETTER(SetterName, ALL(Type), fName, MOVE)              \
    DECL_GETTER(GetterName, ALL(Type), fName)
#endif

// #ifndef DECL_ACCESSOR_M
// #define DECL_ACCESSOR_M(GetterName, SetterName, Type, fName) \
//     DECL_SETTER_M(SetterName, ALL(Type), fName)  \
//     DECL_GETTER(GetterName, ALL(Type), fName)
// #endif

#ifndef DECL_SETTER_PROXY
#define DECL_SETTER_PROXY(SetterName, Type, Proxy, ProxySetterName)     \
    /* NOLINTNEXTLINE */                                                \
    inline void SetterName(Type&& _set_value) {                         \
        /* NOLINTNEXTLINE */                                            \
        this->Proxy->ProxySetterName(std::forward<Type&&>(_set_value)); \
    }
#endif /* ifndef DECL_SETTER_PROXY(cName, fName, fType) */

#ifndef DECL_GETTER_PROXY
#define DECL_GETTER_PROXY(GetterName, Type, Proxy, ProxyGetterName) \
    inline Type& GetterName() { return this->Proxy->ProxyGetterName(); }
#endif /* ifndef DECL_GETTER_PROXY */

#ifndef DECL_ACCESSOR_PROXY
#define DECL_ACCESSOR_PROXY(SetterName, GetterName, Type, Proxy, ProxySetterName, ProxyGetterName) \
    DECL_SETTER_PROXY(SetterName, ALL(Type), Proxy, ProxySetterName)                               \
    DECL_GETTER_PROXY(GetterName, ALL(Type), Proxy, ProxyGetterName)
#endif

#ifndef DECL_ACCESSOR_PROXY_S
#define DECL_ACCESSOR_PROXY_S(SetterName, GetterName, Type, Proxy, ProxySetterGetterName) \
    DECL_SETTER_PROXY(SetterName, ALL(Type), Proxy, ProxySetterGetterName)                \
    DECL_GETTER_PROXY(GetterName, ALL(Type), Proxy, ProxySetterGetterName)
#endif

#ifndef DECL_ACCESSOR_PROXY_SS
#define DECL_ACCESSOR_PROXY_SS(SetterGetterName, Type, Proxy, ProxySetterGetterName) \
    DECL_SETTER_PROXY(SetterGetterName, ALL(Type), Proxy, ProxySetterGetterName)     \
    DECL_GETTER_PROXY(SetterGetterName, ALL(Type), Proxy, ProxySetterGetterName)
#endif

#ifndef DECL_ACCESSOR_PROXY_SSS
#define DECL_ACCESSOR_PROXY_SSS(SetterGetterName, Type, Proxy, MOVE)          \
    DECL_SETTER_PROXY(SetterGetterName, ALL(Type), Proxy, SetterGetterName, ) \
    DECL_GETTER_PROXY(SetterGetterName, ALL(Type), Proxy, SetterGetterName)
#endif

#define FRAMEWORK_STATUS_MACROS_CONCAT_NAME(x, y) FRAMEWORK_STATUS_MACROS_CONCAT_IMPL(x, y)
#define FRAMEWORK_STATUS_MACROS_CONCAT_IMPL(x, y) x##y
#define FRAMEWORK_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
    auto(statusor) = (rexpr);                                 \
    if ((statusor).isErr()) {                                 \
        return Err((statusor).unwrapErr());                   \
    }                                                         \
    (lhs) = std::move((statusor).unwrap())

#define FRAMEWORK_ASSIGN_OR_RETURN(lhs, rexpr) \
    FRAMEWORK_ASSIGN_OR_RETURN_IMPL(FRAMEWORK_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define FRAMEWORK_RETURN_IF_ERROR(...)  \
    do {                                \
        auto _r = (__VA_ARGS__);        \
        if (_r.isErr()) {               \
            return Err(_r.unwrapErr()); \
        }                               \
    } while (0)

#endif /* ifndef FRAMEWORK_UTIL_H */
