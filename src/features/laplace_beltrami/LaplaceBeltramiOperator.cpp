#include "LaplaceBeltramiOperator.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

namespace lr
{

std::vector<float> LaplaceBeltramiOperator::calculateMagnitude(std::span<const glm::vec3> positions,
                                                               std::span<const glm::uvec3> triangles)
{
    const Eigen::Index vertexCount = static_cast<Eigen::Index>(positions.size());
    Eigen::MatrixXd    vertices(vertexCount, 3);
    for (Eigen::Index i = 0; i < vertexCount; ++i)
    {
        vertices.row(i) << positions[static_cast<size_t>(i)].x, positions[static_cast<size_t>(i)].y,
            positions[static_cast<size_t>(i)].z;
    }

    Eigen::MatrixXi faces(static_cast<Eigen::Index>(triangles.size()), 3);
    for (Eigen::Index f = 0; f < faces.rows(); ++f)
    {
        const glm::uvec3 triangle = triangles[static_cast<size_t>(f)];
        faces(f, 0)               = static_cast<int>(triangle.x);
        faces(f, 1)               = static_cast<int>(triangle.y);
        faces(f, 2)               = static_cast<int>(triangle.z);
    }

    Eigen::SparseMatrix<double> cotangentMatrix;
    Eigen::SparseMatrix<double> massMatrix;
    igl::cotmatrix(vertices, faces, cotangentMatrix);
    igl::massmatrix(vertices, faces, igl::MASSMATRIX_TYPE_VORONOI, massMatrix);

    const Eigen::MatrixXd laplaceBeltrami =
        massMatrix.diagonal().cwiseMax(1e-12).cwiseInverse().asDiagonal() * cotangentMatrix * vertices;

    std::vector<float> magnitudes(static_cast<size_t>(vertexCount));
    for (Eigen::Index i = 0; i < vertexCount; ++i)
    {
        magnitudes[static_cast<size_t>(i)] = static_cast<float>(laplaceBeltrami.row(i).norm());
    }
    return magnitudes;
}

} // namespace lr
