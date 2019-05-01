#include <algorithm>
#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;

KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    KNNClassifier::n_neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y) {
    // We should apply PCA here
    KNNClassifier::X = X;
    KNNClassifier::y = y;
}

vector<int> KNNClassifier::nSortedIndexes(Vector v, unsigned int n) {
    vector<int> sortedIndexes(v.size(), 0);
    for (int i = 0; i != sortedIndexes.size(); i++) {
        sortedIndexes[i] = i;
    }
    sort(sortedIndexes.begin(), sortedIndexes.end(),
         [&](const int &a, const int &b) {
             return (v[a] > v[b]);
         }
    );
    sortedIndexes.resize(n);
    sortedIndexes.shrink_to_fit();
    return sortedIndexes;
}

bool KNNClassifier::mostAppearingValue(std::vector<int> &sortedIndexes, Matrix &values) {
    int pos = 0;
    int neg = 0;

    for (auto it = sortedIndexes.begin(); it < sortedIndexes.end(); ++it) {
        if (values(*it, 0) == true) { ++pos; } //supuse que la matriz y es una columna
        if (values(*it, 0) == false) { ++neg; } //supuse que la matriz y es una columna
    }
    if (pos >= neg) { // Por default, si hay empate, la tomamos como positiva. Se puede rever para testear
        return true; //true es positivo.
    } else {
        return false;
    }
}


bool KNNClassifier::predict_row(Vector row) {
    SparseMatrix A = KNNClassifier::X - Vector(KNNClassifier::X.rows(), 1) * row.transpose();
    A = A.cwiseProduct(A); // This should be like multiplying each element by itself
    Vector sums = A * Vector(KNNClassifier::X.cols(), 1); // This is like summing all rows

    // We get the n_neighbors indexes with lowest values
    vector<int> sortedIndexes = KNNClassifier::nSortedIndexes(sums, KNNClassifier::n_neighbors);
    return KNNClassifier::mostAppearingValue(sortedIndexes, KNNClassifier::y);
}

Vector KNNClassifier::predict(SparseMatrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        ret(k) = this->predict_row(X.row(k));
    }

    return ret;
}
