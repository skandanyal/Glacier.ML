#ifndef LOGS_HPP
#define LOGS_HPP

#include <iostream>
#include <cstdlib>

// 1. Errors: Always ON (Safety)
#define LOG_ERROR(x) do { std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE); } while(0)

// 2. Info/Time: ON for Debug and Release, OFF for Benchmark
#if defined(GLACIER_DEBUG) || defined(GLACIER_RELEASE)
    #define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n"
#else
    #define LOG_INFO(x)
#endif

// 3. Debug/Update: ON only for Debug
#ifdef GLACIER_DEBUG
    #define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val << "\n"
    #define LOG_UPDATE(x) std::cout << "\033[35m[DEBUG] \033[0m" << x << "\n"
#else
    #define LOG_DEBUG(x, x_val)
    #define LOG_UPDATE(x)
#endif

#endif //LOGS_HPP