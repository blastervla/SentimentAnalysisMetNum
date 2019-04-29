#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <map>
#include <set>

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

template<typename A, typename B>
std::pair<B,A> flip_pair(const std::pair<A,B> &p)
{
    return std::pair<B,A>(p.second, p.first);
}

template<typename A, typename B>
std::multimap<B,A> flip_map(const std::map<A,B> &src)
{
    std::multimap<B,A> dst;
    std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
                   flip_pair<A,B>);
    return dst;
}

double KNNClassifier::mostAppearingValue(std::vector<int> &sortedIndexes, Matrix &values) {
    map<double, int> valueApparitions;
    for (auto i = sortedIndexes.begin(); i < sortedIndexes.end(); i++) {
        double currentValue = values.coeff(0, 1);

        auto val = valueApparitions.find(currentValue);
        if (val == valueApparitions.end()) {
            // Add count
            val->second = 1;
        } else {
            val->second++;
        }
    }

    return flip_map(valueApparitions).end()->second; // We flip map so that it's ordered in ascending order by value.
}

double KNNClassifier::predict_row(Vector row) {
    SparseMatrix X = KNNClassifier::X - Matrix::Identity(KNNClassifier::X.rows(), KNNClassifier::X.cols()) * row;
    X = X.cwiseProduct(X); // This should be like multiplying each element by itself
    Vector sums = X * Vector(KNNClassifier::X.cols(), 1); // This is like summing all rows

    // We get the n_neighbors indexes with lowest values
    vector<int> sortedIndexes = KNNClassifier::nSortedIndexes(sums, KNNClassifier::n_neighbors);
    return KNNClassifier::mostAppearingValue(sortedIndexes, KNNClassifier::y);
}

Vector KNNClassifier::predict(SparseMatrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        ret(k) = 0;
    }

    return ret;
}
