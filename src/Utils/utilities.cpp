#include "Glacier/Utils/utilities.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

void Glacier::Utils::read_csv_1d(const std::string& filename, std::vector<float>& x, std::vector<float>& y,
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
void Glacier::Utils::read_csv_r(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& targets,
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

void Glacier::Utils::read_csv_c(const std::string& filename, std::vector<std::vector<float>>& x_train, std::vector<std::string>& y_train,
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

float Glacier::Utils::mean(const std::vector<float>& x) {
    float mean = 0.0;
    for (auto i:x) {
        mean += i;
    }
    return mean / x.size();
}

double Glacier::Utils::get_p_value(double t_stat, int dof) {
    using namespace boost::math;

    students_t dist(dof);
    double p = 2.0 * (1.0 - cdf(dist, std::abs(t_stat)));
    return p;
}

/**
 * Generic evaluation for binary classification.
 * Decoupled from the model class to support Systems-level modularity.
 */

void analyze_performance(const std::vector<std::string> &y_test, const std::vector<std::string> &y_pred) {
    if (y_test.empty() || y_test.size() != y_pred.size()) {
        return;
    }

    // Identify labels dynamically from the test set
    std::string label_pos = y_test[0];
    std::string label_neg = "";

    for (const auto& s : y_test) {
        if (s != label_pos) {
            label_neg = s;
            break;
        }
    }

    float tp = 0, fn = 0, fp = 0, tn = 0;

    for (size_t i = 0; i < y_test.size(); i++) {
        if (y_test[i] == label_pos) {
            if (y_pred[i] == label_pos) tp++;
            else fn++;
        } else {
            if (y_pred[i] == label_pos) fp++;
            else tn++;
        }
    }

    // Output Confusion Matrix
    std::cout << "--- Confusion Matrix ---\n";
    std::cout << "Target: " << label_pos << " (Pos) vs " << label_neg << " (Neg)\n";
    std::cout << "TP: " << tp << " | FN: " << fn << "\n";
    std::cout << "FP: " << fp << " | TN: " << tn << "\n\n";

    // Metrics Calculation
    float accuracy = (tp + tn) / y_test.size();
    float recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    float precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    float f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    std::cout << "--- Metrics ---\n";
    std::cout << "Accuracy:  " << accuracy << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall:    " << recall << "\n";
    std::cout << "F1 Score:  " << f1 << "\n";
}
