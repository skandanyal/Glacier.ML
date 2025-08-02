#ifndef UTILITIES_H
#define UTILITIES_H

namespace Utils{
    void read_csv_1d(const std::string& filename, std::vector<float>& x, std::vector<float>& y,
        bool has_header = true);
    void read_csv(const std::string& filename, std::vector<std::vector<float>>& x, std::vector<float>& y,
        bool has_header = true);
    void read_csv(const std::string& filename, std::vector<std::vector<float>>& x, std::vector<std::string>& y,
        bool has_header = true);
    float mean(std::vector<float> x);
    double get_p_value(double t_stat, int dof);
};

#endif //UTILITIES_H
