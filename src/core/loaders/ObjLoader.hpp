#pragma once

#include "Loader.hpp"

namespace lr
{

class ObjLoader final : public Loader
{
public:
	MeshSequence load(const std::filesystem::path &path,
						 const MeshLayout &layout) const override;
};

}  // namespace lr
