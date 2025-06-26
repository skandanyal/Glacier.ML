// matrix multiplication of different dimensions
#include<iostream>
#include<Eigen/Dense>

using Eigen::Matrix;

int main() {
	Matrix<float, 1,3> mat1;
	Matrix<float, 3,2> mat2;
	Matrix<float, 2,1> mat3;

	mat1 << 3, 4, 5;
	mat2 << 1, 2,
			4, 5,
			7, 8;
	mat3 << 10,
			11;

	try {
		if(mat1.cols() != mat2.rows() || mat2.cols() != mat3.rows()) {
			throw std::runtime_error("Matrices are not compatible for multiplication.");
		}
		std::cout << "Multiplication: \n" << mat1 * mat2 * mat3 << "\n";
	} catch(std::runtime_error &e) {
		std::cerr << "Runtime error: " << e.what() << "\n";
	} catch(std::exception &e) {
		std::cerr << "Standard exception: " << e.what() << "\n";
	}
	return 0;

	// Matrix<float, 2,2> mat4;
	// mat4 << 4,9,-9,6;
	// std::cout << mat4.determinant()<< "\n";
	// return 0;
}