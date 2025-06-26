#include<iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;

int main() {
	MatrixXd mat1 = MatrixXd::Random(3,3);
	MatrixXd mat2 = MatrixXd::Random(3,3);

	MatrixXd mat3 = mat1 + mat2;

	cout << mat1 << endl << mat2 << endl << mat3;

	return 0;
}
