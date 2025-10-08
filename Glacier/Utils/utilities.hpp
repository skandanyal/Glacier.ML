#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include <boost/math/distributions/students_t.hpp>

namespace Glacier {
    namespace Utils{
        void read_csv_1d(const std::string& filename, std::vector<float>& x, std::vector<float>& y, bool has_header = true);
        void read_csv(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& targets, bool has_header = true);
        void read_csv(const std::string& filename, std::vector<std::vector<float>>& x, std::vector<std::string>& y, bool has_header = true);
        float mean(const std::vector<float>& x);
        double get_p_value(double t_stat, int dof);
    };
}

inline void Glacier::Utils::read_csv_1d(const std::string& filename, std::vector<float>& x, std::vector<float>& y,
bool has_header ) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    std::string line;
    if (has_header && std::getline(file, line)) {}

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_val, y_val;

        if (std::getline(ss, x_val, ',') && std::getline(ss, y_val, ',')) {
            x.push_back(std::stof(x_val));
            y.push_back(std::stof(y_val));
        }
    }
}

inline void Glacier::Utils::read_csv(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& targets,
              bool has_header) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file.");
    }

    std::string line;
    if (has_header && std::getline(file, line)) {
        // skip header
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }

        // Assume last column is target (y)
        if (!row.empty()) {
            float y = row.back();
            row.pop_back();
            features.push_back(row);
            targets.push_back(y);
        }
    }
}

inline void Glacier::Utils::read_csv(const std::string& filename, std::vector<std::vector<float>>& x_train, std::vector<std::string>& y_train,
                            bool has_header) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    if (has_header && std::getline(file, line)) {
        // Skip header
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        std::vector<std::string> tokens;

        // Read all tokens in the line
        while (std::getline(ss, cell, ',')) {
            tokens.push_back(cell);
        }

        if (!tokens.empty()) {
            std::string label = tokens.back();  // last column as label
            tokens.pop_back(); // remove label from features

            for (const std::string& token : tokens) {
                row.push_back(std::stof(token));
            }

            x_train.push_back(row);
            y_train.push_back(label);
        }
    }
}

inline float Glacier::Utils::mean(const std::vector<float>& x) {
    float mean = 0.0;
    for (auto i:x) {
        mean += i;
    }
    return mean / x.size();
}

inline double Glacier::Utils::get_p_value(double t_stat, int dof) {
    using namespace boost::math;

    students_t dist(dof);
    double p = 2.0 * (1.0 - cdf(dist, std::abs(t_stat)));
    return p;
}


#endif //UTILITIES_HPP
