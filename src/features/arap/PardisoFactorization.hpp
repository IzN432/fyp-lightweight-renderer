#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <mkl_pardiso.h>

#include <array>
#include <vector>

namespace lr
{

// Owns one PARDISO analysis/factorization of a real symmetric positive-definite matrix.
// The factorized matrix is stored as zero-based upper-triangular CSR.
class PardisoFactorization
{
public:
    PardisoFactorization();
    ~PardisoFactorization();

    PardisoFactorization(const PardisoFactorization &)            = delete;
    PardisoFactorization &operator=(const PardisoFactorization &) = delete;

    bool compute(const Eigen::SparseMatrix<double> &matrix);
    bool solve(const Eigen::MatrixXd &rhs, Eigen::MatrixXd &solution);
    void reset() noexcept;

private:
    std::array<void *, 64>   m_handle{};
    std::array<MKL_INT, 64>  m_parameters{};
    std::vector<double>      m_values;
    std::vector<MKL_INT>     m_rowOffsets;
    std::vector<MKL_INT>     m_columnIndices;
    MKL_INT                  m_size       = 0;
    bool                     m_hasInternalState = false;
    bool                     m_factorized = false;
};

} // namespace lr
