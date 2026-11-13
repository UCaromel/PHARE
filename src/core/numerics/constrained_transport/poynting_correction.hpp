#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_POYNTING_CORRECTION_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_POYNTING_CORRECTION_HPP

#include "core/utilities/types.hpp"
#include "core/utilities/index/index.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/constrained_transport/upwind_edge_b.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"

#include <cstdlib>
#include <type_traits>

namespace PHARE::core
{
template<typename GridLayout, template<typename> typename Reconstruction>
class PoyntingCorrection
{
    constexpr static auto dimension = GridLayout::dimension;
    using EdgeB_t                   = UpwindEdgeB<GridLayout, Reconstruction>;

public:
    explicit PoyntingCorrection(GridLayout const& layout)
        : layout_{layout}
    {
    }

    void operator()(auto const& ct_state, auto const& state, auto& fluxes) const
    {
        // Apply Poynting flux correction to energy: ∂E/∂t -= ∇·(E×B)
        // Because E from CT includes resistive (ηJ) and hyper-resistive terms, E×B
        // automatically captures the resistive Poynting flux.
        for_N<dimension>([&](auto i) {
            constexpr auto direction = static_cast<Direction>(i());

            layout_.evalOnBox(fluxes.template expose_centering<direction>(), [&](auto&... indices) {
                auto& F_Etot = fluxes.template get_dir<direction>({indices...}).Etot();
                poynting_energy_flux_<direction>(ct_state, state.E, state.B,
                                                 MeshIndex<dimension>{indices...}, F_Etot);
            });
        });
    }

    // experiment PHARE_CTP_CENTRAL=1: edge B as the plain 2-point average across the third
    // direction (the scripts/ctp.py form FHface/derivation.md S2 assumed), instead of the upwind
    // (reconstructed) edge B
    template<Component edge, Component component>
    static double edgeB_(auto const& ct_state, auto const& B, MeshIndex<dimension> const& idx)
    {
        if (central_)
        {
            constexpr auto d
                = 3 - static_cast<std::size_t>(edge) - static_cast<std::size_t>(component);
            if constexpr (d < dimension)
                return 0.5
                       * (B(component)(idx)
                          + B(component)(
                              GridLayout::template previous<static_cast<Direction>(d)>(idx)));
            else
                return B(component)(idx);
        }
        return EdgeB_t::template at<edge, component>(ct_state, B, idx);
    }
    // not const: PHARE_CTP_CENTRAL_MASK (ctpsurprise) switches it per EMF part in ComputeFluxes
    static inline bool central_ = std::getenv("PHARE_CTP_CENTRAL") != nullptr;
    // experiment PHARE_CTP_CONSISTENT=1 (ctpsurprise/derivation S2 2D, S4 3D): product formed on the face,
    // projected E times the Godunov transverse B (Rusanov average of transverse_reconstruct), so the
    // O(B0) ideal exchange matches the momentum flux's Lorentz work. Not const: the budget probe flips it.
    static inline bool consistent_ = std::getenv("PHARE_CTP_CONSISTENT") != nullptr;

    // public for the closure probe (experiment)
    template<auto direction>
    void poynting_energy_flux_(auto const& ct_state, auto const& E, auto const& B,
                               MeshIndex<dimension> const& index, auto& F_Etot) const
    {
        // Magnetic energy flux through the face: S.n = (E x B).n. Since S_i = eps_ijk E_j B_k,
        // a face never needs its own E component, which leaves exactly two products per face.

        // Same as GridLayout::project, but for an expression rather than a stored field: the
        // integrand is a product of two fields, and since both are point values the product
        // must be formed on the edge before being projected onto the face. Projecting the
        // factors separately and multiplying on the face would be a different (and, beyond
        // order 2, wrong) discretisation.
        auto project_on_face = [&](auto const& stencil, auto&& integrand) {
            std::decay_t<decltype(integrand(index))> result = 0.;

            for (auto const& wp : stencil)
                result += wp.coef * integrand(index + wp.indexes);

            return result;
        };

        auto const& Ex = E(Component::X);
        auto const& Ey = E(Component::Y);
        auto const& Ez = E(Component::Z);

        using implT = GridLayout::implT;

        if constexpr (dimension >= 2)
            if (consistent_ && !central_)
            {
                using Rtor     = Reconstructor<Reconstruction<GridLayout>>;
                auto const val = [](auto const& f) { return [&f](auto const& i) { return f(i); }; };
                if constexpr (direction == Direction::X)
                {
                    auto const [BL, BR] = Rtor::template transverse_reconstruct<Direction::X>(B, index);
                    double const gy     = 0.5 * (BL.y + BR.y);
                    double const gz     = 0.5 * (BL.z + BR.z);
                    double const ey     = project_on_face(implT::edgeYToFaceX(), val(Ey));
                    double const ez     = project_on_face(implT::edgeZToFaceX(), val(Ez));
                    F_Etot += ey * gz - ez * gy;
                }
                else if constexpr (direction == Direction::Y)
                {
                    auto const [BL, BR] = Rtor::template transverse_reconstruct<Direction::Y>(B, index);
                    double const gx     = 0.5 * (BL.x + BR.x);
                    double const gz     = 0.5 * (BL.z + BR.z);
                    double const ez     = project_on_face(implT::edgeZToFaceY(), val(Ez));
                    double const ex     = project_on_face(implT::edgeXToFaceY(), val(Ex));
                    F_Etot += ez * gx - ex * gz;
                }
                else if constexpr (direction == Direction::Z)
                {
                    auto const [BL, BR] = Rtor::template transverse_reconstruct<Direction::Z>(B, index);
                    double const gx     = 0.5 * (BL.x + BR.x);
                    double const gy     = 0.5 * (BL.y + BR.y);
                    double const ex     = project_on_face(implT::edgeXToFaceZ(), val(Ex));
                    double const ey     = project_on_face(implT::edgeYToFaceZ(), val(Ey));
                    F_Etot += ex * gy - ey * gx;
                }
                return;
            }

        if constexpr (direction == Direction::X)
        {
            constexpr auto zEdgeToXFace = implT::edgeZToFaceX();
            constexpr auto yEdgeToXFace = implT::edgeYToFaceX();

            auto const EzBy = project_on_face(zEdgeToXFace, [&](auto const& i) {
                return Ez(i) * edgeB_<Component::Z, Component::Y>(ct_state, B, i);
            });
            auto const EyBz = project_on_face(yEdgeToXFace, [&](auto const& i) {
                return Ey(i) * edgeB_<Component::Y, Component::Z>(ct_state, B, i);
            });

            F_Etot += EyBz - EzBy;
        }
        else if constexpr (direction == Direction::Y)
        {
            constexpr auto zEdgeToYFace = implT::edgeZToFaceY();
            constexpr auto xEdgeToYFace = implT::edgeXToFaceY();

            auto const EzBx = project_on_face(zEdgeToYFace, [&](auto const& i) {
                return Ez(i) * edgeB_<Component::Z, Component::X>(ct_state, B, i);
            });
            auto const ExBz = project_on_face(xEdgeToYFace, [&](auto const& i) {
                return Ex(i) * edgeB_<Component::X, Component::Z>(ct_state, B, i);
            });

            F_Etot += EzBx - ExBz;
        }
        else if constexpr (direction == Direction::Z)
        {
            constexpr auto xEdgeToZFace = implT::edgeXToFaceZ();
            constexpr auto yEdgeToZFace = implT::edgeYToFaceZ();

            auto const ExBy = project_on_face(xEdgeToZFace, [&](auto const& i) {
                return Ex(i) * edgeB_<Component::X, Component::Y>(ct_state, B, i);
            });
            auto const EyBx = project_on_face(yEdgeToZFace, [&](auto const& i) {
                return Ey(i) * edgeB_<Component::Y, Component::X>(ct_state, B, i);
            });

            F_Etot += ExBy - EyBx;
        }
    }

private:
    GridLayout layout_;
};

} // namespace PHARE::core

#endif
