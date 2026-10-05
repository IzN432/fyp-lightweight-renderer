#include "PardisoFactorization.hpp"

#include <limits>

namespace lr
{
namespace
{

constexpr MKL_INT matrixType = 2; // Real symmetric positive definite.
constexpr MKL_INT maxFactors = 1;
constexpr MKL_INT matrixNumber = 1;
constexpr MKL_INT messageLevel = 0;

} // namespace

PardisoFactorization::PardisoFactorization()
{
    static_assert(sizeof(MKL_INT) == sizeof(int), "The ARAP backend requires oneMKL's LP64 interface");
    pardisoinit(m_handle.data(), &matrixType, m_parameters.data());
    m_parameters[34] = 1; // Eigen and the stored CSR arrays use zero-based indexing.
}

PardisoFactorization::~PardisoFactorization()
{
    reset();
}

bool PardisoFactorization::compute(const Eigen::SparseMatrix<double> &matrix)
{
    reset();
    if (matrix.rows() <= 0 || matrix.rows() != matrix.cols() ||
        matrix.rows() > std::numeric_limits<MKL_INT>::max())
    {
        return false;
    }

    using RowMajorMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    RowMajorMatrix rowMajor = matrix;
    rowMajor.makeCompressed();

    m_size = static_cast<MKL_INT>(rowMajor.rows());
    m_rowOffsets.reserve(static_cast<size_t>(m_size) + 1);
    m_rowOffsets.push_back(0);
    for (MKL_INT row = 0; row < m_size; ++row)
    {
        bool hasDiagonal = false;
        for (RowMajorMatrix::InnerIterator entry(rowMajor, row); entry; ++entry)
        {
            const MKL_INT column = static_cast<MKL_INT>(entry.col());
            if (column < row)
            {
                continue;
            }
            hasDiagonal |= column == row;
            m_columnIndices.push_back(column);
            m_values.push_back(entry.value());
        }
        if (!hasDiagonal)
        {
            reset();
            return false;
        }
        m_rowOffsets.push_back(static_cast<MKL_INT>(m_values.size()));
    }

    constexpr MKL_INT phase = 12; // Analysis and numerical factorization.
    constexpr MKL_INT rightHandSides = 3;
    MKL_INT error = 0;
    m_hasInternalState = true;
    pardiso(m_handle.data(), &maxFactors, &matrixNumber, &matrixType, &phase, &m_size,
            m_values.data(), m_rowOffsets.data(), m_columnIndices.data(), nullptr,
            &rightHandSides, m_parameters.data(), &messageLevel, nullptr, nullptr, &error);
    m_factorized = error == 0;
    if (!m_factorized)
    {
        reset();
    }
    return m_factorized;
}

bool PardisoFactorization::solve(const Eigen::MatrixXd &rhs, Eigen::MatrixXd &solution)
{
    if (!m_factorized || rhs.rows() != m_size || rhs.cols() <= 0 ||
        rhs.cols() > std::numeric_limits<MKL_INT>::max())
    {
        return false;
    }

    Eigen::MatrixXd mutableRhs = rhs;
    solution.resize(rhs.rows(), rhs.cols());

    constexpr MKL_INT phase = 33;
    const MKL_INT rightHandSides = static_cast<MKL_INT>(rhs.cols());
    MKL_INT error = 0;
    pardiso(m_handle.data(), &maxFactors, &matrixNumber, &matrixType, &phase, &m_size,
            m_values.data(), m_rowOffsets.data(), m_columnIndices.data(), nullptr,
            &rightHandSides, m_parameters.data(), &messageLevel, mutableRhs.data(),
            solution.data(), &error);
    return error == 0;
}

void PardisoFactorization::reset() noexcept
{
    if (m_hasInternalState)
    {
        constexpr MKL_INT phase = -1;
        constexpr MKL_INT rightHandSides = 0;
        MKL_INT error = 0;
        pardiso(m_handle.data(), &maxFactors, &matrixNumber, &matrixType, &phase, &m_size,
                m_values.data(), m_rowOffsets.data(), m_columnIndices.data(), nullptr,
                &rightHandSides, m_parameters.data(), &messageLevel, nullptr, nullptr, &error);
    }

    m_handle.fill(nullptr);
    m_parameters.fill(0);
    pardisoinit(m_handle.data(), &matrixType, m_parameters.data());
    m_parameters[34] = 1;
    m_values.clear();
    m_rowOffsets.clear();
    m_columnIndices.clear();
    m_size       = 0;
    m_hasInternalState = false;
    m_factorized = false;
}

} // namespace lr
