#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
private:
    unsigned int n_neighbors;
    SparseMatrix X;
    Matrix y;

    double predict_row(Vector row);
    std::vector<int> nSortedIndexes(Vector v, unsigned int n);
    double mostAppearingValue(std::vector<int> &sortedIndexes, Matrix &values);
};
