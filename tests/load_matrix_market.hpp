#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

Eigen::MatrixXd loadMatrixMarket(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file " + filename);

    std::string line;
    do {
        std::getline(file, line);
    } while (!file.eof() && line[0] == '%');

    std::istringstream header(line);
    int rows, cols, entries;
    header >> rows >> cols >> entries;

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(rows, cols);
    int r, c;
    double val;
    for (int i = 0; i < entries; ++i) {
        file >> r >> c >> val;
        M(r - 1, c - 1) = val; 
    }
    return M;
}
