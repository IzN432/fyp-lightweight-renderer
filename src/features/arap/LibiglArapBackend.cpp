#include "LibiglArapBackend.hpp"

namespace lr
{

bool LibiglArapBackend::precompute(const Eigen::MatrixXd &restPositions, const Eigen::MatrixXi &faces,
                                   const Eigen::VectorXi &constrainedIndices)
{
    return igl::arap_precomputation(restPositions, faces, 3, constrainedIndices, m_data);
}

bool LibiglArapBackend::solve(const Eigen::MatrixXd &constraintTargets, Eigen::MatrixXd &positions,
                              int iterations)
{
    m_data.max_iter = iterations;
    return igl::arap_solve(constraintTargets, m_data, positions);
}

} // namespace lr
