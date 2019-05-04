#pragma once

#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(SparseMatrix X);

    Eigen::MatrixXd transform(SparseMatrix X);

private:
    MatrixXd base;
    unsigned int n_components;
};
