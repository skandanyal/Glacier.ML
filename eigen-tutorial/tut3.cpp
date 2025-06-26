#include <iostream>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using namespace std;

//dimensions are set during compile time
int main() {
	Matrix3d m = Matrix3d::Random();
	m = (m + Eigen::MatrixXd::Constant(3,3,1.2)) * 50;
	std::cout << "m =" << std::endl << m << std::endl;
	Vector3d v(3);
	v << 1, 2, 3;
	std::cout << "v =" << std::endl << v << std::endl;
	std::cout << "m * v =" << std::endl << m * v << std::endl;
}