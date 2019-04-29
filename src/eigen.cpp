#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix &X, unsigned num_iter, double eps) {
    Vector b = Vector::Random(X.cols());
    double eigenvalue;

    for (int i = 0; i < num_iter; i++) {
        b = X * b / (X * b).squaredNorm();
    }

    eigenvalue =
            (b.transpose() * X * b)(0) / b.norm(); // (0) gets the scalar value as multiplication is a 1x1 matrix
    return make_pair(eigenvalue, b / b.norm());
}

void deinflate(Matrix &X) {
    Vector u(X.rows(), 0);
    u[0] = X.col(0).norm();
    Matrix H = Matrix::Identity(X.rows(), X.cols()) - (2 * u * u.transpose()) / u.squaredNorm();
    X = H * X;
    X = X.block(1, 1, X.rows() - 1, X.cols() - 1); // Get submatrix eliminating first column and row.
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix &X, unsigned num, unsigned num_iter, double epsilon) {
    Matrix A(X);
    Vector eigvalues(num); // Vector con autovalores
    Matrix eigvectors(A.rows(), num); // Matriz con autovectores como columnas

    for (int i = 0; i < num; i++) {
        pair<double, Vector> res = power_iteration(A, num_iter, epsilon);
        eigvalues[i] = res.first;
        eigvectors.col(i) = res.second;
        deinflate(A);
    }

    return make_pair(eigvalues, eigvectors);
}
