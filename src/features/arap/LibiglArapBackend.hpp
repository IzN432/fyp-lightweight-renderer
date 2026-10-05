#pragma once

#include "ArapBackend.hpp"

#include <cassert>
#include <igl/arap.h>

namespace lr
{

class LibiglArapBackend final : public ArapBackend
{
public:
    std::string_view name() const override { return "libigl / Eigen"; }
    bool precompute(const Eigen::MatrixXd &restPositions, const Eigen::MatrixXi &faces,
                    const Eigen::VectorXi &constrainedIndices) override;
    bool solve(const Eigen::MatrixXd &constraintTargets, Eigen::MatrixXd &positions,
               int iterations) override;

private:
    igl::ARAPData m_data;
};

} // namespace lr
