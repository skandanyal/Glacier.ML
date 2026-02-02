#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <string>

namespace Glacier {
    namespace Utils{
        void read_csv_1d(const std::string& filename,
            std::vector<float>& x,
            std::vector<float>& y,
            bool has_header = true
            );

        void read_csv_r(const std::string& filename,
            std::vector<std::vector<float>>& features,
            std::vector<float>& targets,
            bool has_header = true
            );

        void read_csv_c(const std::string& filename,
            std::vector<std::vector<float>>& x,
            std::vector<std::string>& y,
            bool has_header = true
            );

        float mean(const std::vector<float>& x);

        double get_p_value(double t_stat, int dof);

        void binary_classification_report(std::vector<std::vector<float>> &x_test,
            std::vector<std::string> &y_test
            );
    };
}

#endif //UTILITIES_HPP
