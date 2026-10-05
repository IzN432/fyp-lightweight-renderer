#include "PardisoArapBackend.hpp"

#include <cassert>
#include <igl/ARAPEnergyType.h>
#include <igl/arap_rhs.h>
#include <igl/columnize.h>
#include <igl/cotmatrix.h>
#include <igl/covariance_scatter_matrix.h>
#include <igl/fit_rotations.h>
#include <igl/slice.h>

#include <vector>

namespace lr
{

bool PardisoArapBackend::precompute(const Eigen::MatrixXd &restPositions, const Eigen::MatrixXi &faces,
                                    const Eigen::VectorXi &constrainedIndices)
{
    m_vertexCount = restPositions.rows();
    m_fixed       = constrainedIndices;
    if (m_vertexCount <= 0 || restPositions.cols() != 3 || faces.cols() != 3 || m_fixed.size() == 0)
    {
        return false;
    }

    std::vector<bool> fixedMask(static_cast<size_t>(m_vertexCount), false);
    for (Eigen::Index i = 0; i < m_fixed.size(); ++i)
    {
        const int index = m_fixed(i);
        if (index < 0 || index >= m_vertexCount || fixedMask[static_cast<size_t>(index)])
        {
            return false;
        }
        fixedMask[static_cast<size_t>(index)] = true;
    }

    m_free.resize(m_vertexCount - m_fixed.size());
    Eigen::Index freeIndex = 0;
    for (Eigen::Index vertex = 0; vertex < m_vertexCount; ++vertex)
    {
        if (!fixedMask[static_cast<size_t>(vertex)])
        {
            m_free(freeIndex++) = static_cast<int>(vertex);
        }
    }
    if (m_free.size() == 0)
    {
        return false;
    }

    constexpr igl::ARAPEnergyType energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
    Eigen::SparseMatrix<double> laplacian;
    igl::cotmatrix(restPositions, faces, laplacian);
    igl::covariance_scatter_matrix(restPositions, faces, energy, m_covarianceScatter);
    igl::arap_rhs(restPositions, faces, 3, energy, m_arapRhs);

    const Eigen::SparseMatrix<double> quadratic = -laplacian;
    Eigen::SparseMatrix<double> freeFree;
    igl::slice(quadratic, m_free, m_free, freeFree);
    igl::slice(quadratic, m_free, m_fixed, m_freeFixed);
    return m_factorization.compute(freeFree);
}

bool PardisoArapBackend::solve(const Eigen::MatrixXd &constraintTargets, Eigen::MatrixXd &positions,
                               int iterations)
{
    if (constraintTargets.rows() != m_fixed.size() || constraintTargets.cols() != 3 ||
        positions.rows() != m_vertexCount || positions.cols() != 3 || iterations < 0)
    {
        return false;
    }

    const Eigen::Index rotationCount = m_arapRhs.cols() / 9;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        for (Eigen::Index i = 0; i < m_fixed.size(); ++i)
        {
            positions.row(m_fixed(i)) = constraintTargets.row(i);
        }

        Eigen::MatrixXd covariance = m_covarianceScatter * positions.replicate(3, 1);
        covariance /= covariance.array().abs().maxCoeff();

        Eigen::MatrixXd rotations(3, m_covarianceScatter.rows());
        igl::fit_rotations(covariance, true, rotations);

        Eigen::VectorXd columnizedRotations;
        igl::columnize(rotations, rotationCount, 2, columnizedRotations);
        const Eigen::VectorXd linearTerm = -m_arapRhs * columnizedRotations;

        Eigen::MatrixXd rhs(m_free.size(), 3);
        for (Eigen::Index coordinate = 0; coordinate < 3; ++coordinate)
        {
            for (Eigen::Index i = 0; i < m_free.size(); ++i)
            {
                rhs(i, coordinate) = -linearTerm(coordinate * m_vertexCount + m_free(i));
            }
        }
        rhs.noalias() -= m_freeFixed * constraintTargets;

        Eigen::MatrixXd freePositions;
        if (!m_factorization.solve(rhs, freePositions))
        {
            return false;
        }
        for (Eigen::Index i = 0; i < m_free.size(); ++i)
        {
            positions.row(m_free(i)) = freePositions.row(i);
        }
    }
    return true;
}

} // namespace lr
