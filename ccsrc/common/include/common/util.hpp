#ifndef FRAMEWORK_UTIL_H
#define FRAMEWORK_UTIL_H

#define ALL(...) __VA_ARGS__

#ifndef DECL_SETTER
#define DECL_SETTER(SetterName, Type, fieldName, MOVE_OR_COPY) \
    DECL_SETTER_##MOVE_OR_COPY(SetterName, ALL(Type), fieldName)
#endif

#ifndef DECL_SETTER_M
#define DECL_SETTER_M(SetterName, Type, fieldName) \
    /* NOLINTNEXTLINE */                           \
    inline void SetterName(const Type&(fieldName)) { this->fieldName = std::move(fieldName); }
#endif
#ifndef DECL_SETTER_C
#define DECL_SETTER_C(SetterName, Type, fieldName) \
    /* NOLINTNEXTLINE */                           \
    inline void SetterName(const Type&(fieldName)) { this->fieldName = fieldName; }
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

#ifndef DECL_SETTER_PROXY
#define DECL_SETTER_PROXY(SetterName, Type, Proxy, ProxySetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY_##MOVE_OR_COPY(SetterName, ALL(Type), Proxy, ProxySetterName)
#endif

#ifndef DECL_SETTER_PROXY_M
#define DECL_SETTER_PROXY_M(SetterName, Type, Proxy, ProxySetterName) \
    /* NOLINTNEXTLINE */                                              \
    inline void SetterName(const Type& _set_value) {                  \
        /* NOLINTNEXTLINE */                                          \
        this->Proxy->ProxySetterName(std::move(_set_value));          \
    }
#endif

#ifndef DECL_SETTER_PROXY_C
#define DECL_SETTER_PROXY_C(SetterName, Type, Proxy, ProxySetterName) \
    /* NOLINTNEXTLINE */                                              \
    inline void SetterName(const Type& _set_value) {                  \
        /* NOLINTNEXTLINE */                                          \
        this->Proxy->ProxySetterName(_set_value);                     \
    }
#endif

#ifndef DECL_GETTER_PROXY
#define DECL_GETTER_PROXY(GetterName, Type, Proxy, ProxyGetterName) \
    inline Type& GetterName() { return this->Proxy->ProxyGetterName(); }
#endif

#ifndef DECL_ACCESSOR_PROXY
#define DECL_ACCESSOR_PROXY(SetterName, GetterName, Type, Proxy, ProxySetterName, ProxyGetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY(SetterName, ALL(Type), Proxy, ProxySetterName, MOVE_OR_COPY)                               \
    DECL_GETTER_PROXY(GetterName, ALL(Type), Proxy, ProxyGetterName)
#endif

#ifndef DECL_ACCESSOR_PROXY_S
#define DECL_ACCESSOR_PROXY_S(SetterName, GetterName, Type, Proxy, ProxySetterGetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY(SetterName, ALL(Type), Proxy, ProxySetterGetterName, MOVE_OR_COPY)                \
    DECL_GETTER_PROXY(GetterName, ALL(Type), Proxy, ProxySetterGetterName)
#endif

#ifndef DECL_ACCESSOR_PROXY_SS
#define DECL_ACCESSOR_PROXY_SS(SetterGetterName, Type, Proxy, ProxySetterGetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY(SetterGetterName, ALL(Type), Proxy, ProxySetterGetterName, MOVE_OR_COPY)     \
    DECL_GETTER_PROXY(SetterGetterName, ALL(Type), Proxy, ProxySetterGetterName)
#endif

#ifndef DECL_ACCESSOR_PROXY_SSS
#define DECL_ACCESSOR_PROXY_SSS(SetterGetterName, Type, Proxy, MOVE_OR_COPY)              \
    DECL_SETTER_PROXY(SetterGetterName, ALL(Type), Proxy, SetterGetterName, MOVE_OR_COPY) \
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
