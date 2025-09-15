//
// Created by skandan-c-y on 9/3/25.
//

#ifndef LOGS_HPP
#define LOGS_HPP

// errors and exits
#define LOG_ERROR(x) std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);

// high level info while users are using it
#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";

// time taken
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " seconds. \n";

// deeper info to be used during development
#if DEBUG_MODE
    #define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n";
    #define LOG_UPDATE(x) std::cout << "\033[35m[DEBUG] \033[0m" << x << "\n";
#else
    #define LOG_DEBUG(x, x_val);
#endif

#endif LOGS_HPP
