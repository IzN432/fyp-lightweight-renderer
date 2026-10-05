#pragma once

#include <Eigen/Core>

#include <memory>
#include <string_view>

namespace lr
{

// Eigen-level ARAP implementation boundary. ArapSolver owns conversion between the engine's
// Mesh/GLM types and these backend-neutral matrices.
class ArapBackend
{
public:
    virtual ~ArapBackend() = default;

    virtual std::string_view name() const = 0;
    virtual bool precompute(const Eigen::MatrixXd &restPositions, const Eigen::MatrixXi &faces,
                            const Eigen::VectorXi &constrainedIndices) = 0;
    virtual bool solve(const Eigen::MatrixXd &constraintTargets, Eigen::MatrixXd &positions,
                       int iterations) = 0;
};

std::unique_ptr<ArapBackend> createArapBackend();

} // namespace lr
