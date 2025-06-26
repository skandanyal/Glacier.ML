#include<iostream>
#include<Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// dimensions are set at run time

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrixXd;

int main() {
	MatrixXd m = MatrixXd::Random(3,3);
	m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
	std::cout << "m =" << std::endl << m << std::endl;
	VectorXd v(3);
	v << 1, 2, 3;
	std::cout << "v =" << std::endl << v << std::endl;
	std::cout << "m * v =" << std::endl << m * v << std::endl;
}