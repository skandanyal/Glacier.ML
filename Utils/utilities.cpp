#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "utilities.h"
#include <boost/math/distributions/students_t.hpp>

void Utils::read_csv_1d(const std::string& filename, std::vector<float>& x, std::vector<float>& y,
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

float Utils::mean(std::vector<float> x) {
    float mean = 0.0;
    for (auto i:x) {
        mean += i;
    }
    return mean / x.size();
}

double Utils::get_p_value(double t_stat, int dof) {
    using namespace boost::math;

    students_t dist(dof);
    double p = 2.0 * (1.0 - cdf(dist, std::abs(t_stat)));
    return p;
}
