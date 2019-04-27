#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
    KNNClassifier::n_neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
    // We should apply PCA here
    KNNClassifier::X = X;
    KNNClassifier::X = y;
}

vector<int> KNNClassifier::nSortedIndexes(Vector v, unsigned int n) {
    vector<int> sortedIndexes(v.size(), 0);
    for (int i = 0 ; i != sortedIndexes.size() ; i++) {
        sortedIndexes[i] = i;
    }
    sort(sortedIndexes.begin(), sortedIndexes.end(),
         [&](const int& a, const int& b) {
             return (v[a] > v[b]);
         }
    );
    sortedIndexes.resize(n);
    sortedIndexes.shrink_to_fit();
    return sortedIndexes;
}

double KNNClassifier::predict_row(Vector row) {
    SparseMatrix X = KNNClassifier::X - Matrix::Identity(KNNClassifier::X.rows(), KNNClassifier::X.cols()) * row;
    X = X.cwiseProduct(X); // This should be like multiplying each element by itself
    Vector sums = X * Vector(KNNClassifier::X.cols(), 1); // This is like summing all rows

    // We get the n_neighbors indexes with lowest values
    vector<int> sortedIndexes = KNNClassifier::nSortedIndexes(sums, KNNClassifier::n_neighbors);
    return /* FIXME! We should return the most appearing value here */1;
}

Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = 0;
    }

    return ret;
}
