#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
	Matrix2d A;
	A << 2, -1,
		 1, 3;

	Vector2d b(5, 6);

	Vector2d x = A.inverse() * b;

	cout << "Solution x:\n" << x << endl;

	return 0;
}