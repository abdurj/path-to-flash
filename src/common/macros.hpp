#pragma once

#define EVAL_OR_RETURN(expr, ret) \
    ({ \
        auto _status_or = (expr); \
        if (!_status_or.ok()) { \
            return ret; \
        } \
        std::move(_status_or.value()); \
    })

#define EVAL_OR_ASSERT(expr) \
    ({ \
        auto result = (expr); \
        if (!result.ok()) { \
            std::cerr << "Error: " << result.status().message() << std::endl; \
            std::abort(); \
        } \
        std::move(result.value()); \
    })