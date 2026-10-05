#pragma once

#include "ArapBackend.hpp"
#include "PardisoFactorization.hpp"

#include <Eigen/SparseCore>

namespace lr
{

class PardisoArapBackend final : public ArapBackend
{
public:
    std::string_view name() const override { return "libigl ARAP / oneMKL PARDISO"; }
    bool precompute(const Eigen::MatrixXd &restPositions, const Eigen::MatrixXi &faces,
                    const Eigen::VectorXi &constrainedIndices) override;
    bool solve(const Eigen::MatrixXd &constraintTargets, Eigen::MatrixXd &positions,
               int iterations) override;

private:
    Eigen::SparseMatrix<double> m_covarianceScatter;
    Eigen::SparseMatrix<double> m_arapRhs;
    Eigen::SparseMatrix<double> m_freeFixed;
    Eigen::VectorXi             m_fixed;
    Eigen::VectorXi             m_free;
    PardisoFactorization        m_factorization;
    Eigen::Index                m_vertexCount = 0;
};

} // namespace lr
