#include <algorithm>
#include <chrono>
#include <iostream>
#include "knn.h"

#include <pybind11/pybind11.h>

using namespace std;

#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort

KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    this->n_neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y) {
    // We should apply PCA here
    this->X = X;
    this->y = y;
}

vector<int> KNNClassifier::nSortedIndexes(Vector v, unsigned int n) {
    vector<int> sortedIndexes(n, 0);
//    for (int i = 0; i != sortedIndexes.size(); i++) {
//        sortedIndexes[i] = i;
//    }
//    sort(sortedIndexes.begin(), sortedIndexes.end(),
//         [&](const int &a, const int &b) {
//             return (v(a) > v(b));
//         }
//    );
//    sortedIndexes.resize(n);
//    sortedIndexes.shrink_to_fit();

//    for (int i = 0; i < n; ++i) {
//        cout << "At index " << sortedIndexes[i] << ": " << v(sortedIndexes[i]) << endl;
//    }

    // initialize original index locations
    vector<size_t> idx(v.size());
    cout << v.size() << endl;
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});


    int j = 0;
    for (auto i: idx) {
        if (j < 10) {
            cout << "At index " << i << ": " << v(i) << endl;
            sortedIndexes[j] = v(i);
            j++;
        }else {
            break;
        }
    }
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
    SparseMatrix A = X - Vector(X.rows(), 1) * row.transpose();
    A = A.cwiseProduct(A); // This should be like multiplying each element by itself
    Vector sums = A * Vector(X.cols(), 1); // This is like summing all rows

    // We get the n_neighbors indexes with lowest values
    vector<int> sortedIndexes = nSortedIndexes(sums, n_neighbors);
    return mostAppearingValue(sortedIndexes, y);
}

Vector KNNClassifier::predict(SparseMatrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        ret(k) = this->predict_row(X.row(k));
    }

    return ret;
}
