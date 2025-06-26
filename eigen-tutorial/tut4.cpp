#include<iostream>
#include<Eigen/Dense>

using Eigen::Matrix3f;
using Eigen::VectorXf;

int main() {
	Matrix3f ran1 = Matrix3f::Random(3,3);
	Matrix3f ide1 = Matrix3f::Identity(3,3);
	Matrix3f zer1 = Matrix3f::Zero(3,3);

	VectorXf v1(2);
	v1 << 1, 2;

	// std::cout << "Random Matrix: \n" << ran1 << "\n";
	// std::cout << "Identity matrix: \n" << ide1 << "\n";
	// std::cout << "Zero matrix: \n" << zer1 << "\n";

	// std::cout << ran1 * ide1 << "\n";

	try {
		// std::cout << "ran1 * v1: " << ran1 * v1 << "\n";

		if(ran1.cols() != v1.rows()) {
			throw std::runtime_error("Matrix multiplication is incompatible.");
		}
		std::cout << "ran1 * v1: \n" << ran1 * v1;
	} catch (const std::runtime_error &e) {
		std::cerr << "Runtime error: " << e.what() << "\n";
	} catch (const std::exception &e) {
		std::cerr << "Standard Exception: " << e.what() << "\n";
	} catch (...) {
		std::cerr << "Unknown error. \n";
	}
	std::cout << "End.";

	return 0;

}