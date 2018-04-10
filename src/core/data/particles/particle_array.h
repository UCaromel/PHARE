#ifndef PHARE_CORE_DATA_PARTICLES_PARTICLE_ARRAY_H
#define PHARE_CORE_DATA_PARTICLES_PARTICLE_ARRAY_H

#include <cstddef>
#include <vector>

#include "particle.h"

namespace PHARE
{
template<std::size_t Dim>
using ParticleArray = std::vector<Particle<Dim>>;
}


#endif
