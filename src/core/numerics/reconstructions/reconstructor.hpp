#ifndef PHARE_CORE_NUMERICS_RECONSTRUCTIONS_RECONSTRUCTOR_HPP
#define PHARE_CORE_NUMERICS_RECONSTRUCTIONS_RECONSTRUCTOR_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/reconstructions/weno3z.hpp"
#include <type_traits>
#include <utility>

namespace PHARE::core
{
template<typename Reconstruction>
struct Reconstructor
{
public:
    using GridLayout = Reconstruction::GridLayout_t;

    template<auto direction, typename State, typename Field>
    static auto reconstruct_field(State const& S, Field const& F, MeshIndex<Field::dimension> index)
    {
        return reconstruct_with_fallback_<direction>(S, F, index);
    }

    template<auto direction, typename State, typename Field>
    static auto center_reconstruct_field(State const& S, Field const& U,
                                         MeshIndex<Field::dimension> index, auto projection)
    {
        return center_reconstruct_with_fallback_<direction>(S, U, index, projection);
    }

    template<auto direction, typename State>
    static auto reconstruct(State const& S, MeshIndex<GridLayout::dimension> index)
    {
        auto reconstruct_component = [&](auto const& field) {
            return reconstruct_with_fallback_<direction>(S, field, index);
        };

        auto [rhoL, rhoR] = reconstruct_component(S.rho);
        auto [VxL, VxR]   = reconstruct_component(S.V(Component::X));
        auto [VyL, VyR]   = reconstruct_component(S.V(Component::Y));
        auto [VzL, VzR]   = reconstruct_component(S.V(Component::Z));
        auto [PL, PR]     = reconstruct_component(S.P);

        // auto [BL, BR] = center_reconstruct<direction>(S.B, GridLayout::faceXToCellCenter(),
        //                                               GridLayout::faceYToCellCenter(),
        //                                               GridLayout::faceZToCellCenter(), index);

        auto [BL, BR] = transverse_reconstruct<direction>(S, S.B, index);

        PerIndex uL{rhoL, {VxL, VyL, VzL}, BL, PL};
        PerIndex uR{rhoR, {VxR, VyR, VzR}, BR, PR};

        return std::make_pair(uL, uR);
    }

    template<auto direction, typename VecField>
    static auto center_reconstruct(VecField const& U, auto projectionX, auto projectionY,
                                   auto projectionZ, MeshIndex<VecField::dimension> index)
    {
        auto const& Ux = U(Component::X);
        auto const& Uy = U(Component::Y);
        auto const& Uz = U(Component::Z);

        auto [UxL, UxR]
            = Reconstruction::template center_reconstruct<direction>(Ux, index, projectionX);
        auto [UyL, UyR]
            = Reconstruction::template center_reconstruct<direction>(Uy, index, projectionY);
        auto [UzL, UzR]
            = Reconstruction::template center_reconstruct<direction>(Uz, index, projectionZ);

        return std::make_tuple(PerIndexVector{UxL, UyL, UzL}, PerIndexVector{UxR, UyR, UzR});
    }

    template<auto direction>
    static constexpr auto projection()
    {
        if constexpr (direction == Direction::X)
            return GridLayout::faceXToCellCenter();
        else if constexpr (direction == Direction::Y)
            return GridLayout::faceYToCellCenter();
        else if constexpr (direction == Direction::Z)
            return GridLayout::faceZToCellCenter();
    }

    // The normal direction for B is already face centered, so we only reconstruct the transverse
    template<auto direction, typename VecField>
    static auto transverse_reconstruct(VecField const& B, MeshIndex<VecField::dimension> index)
    {
        auto constexpr transverse = []() { // probably should be a util function somewhere
            if constexpr (direction == Direction::X)
                return std::array{Direction::Y, Direction::Z};
            else if constexpr (direction == Direction::Y)
                return std::array{Direction::X, Direction::Z};
            else if constexpr (direction == Direction::Z)
                return std::array{Direction::X, Direction::Y};
        }();


        auto const Bn  = B(static_cast<Component>(direction));
        auto const Bt0 = B(static_cast<Component>(transverse[0]));
        auto const Bt1 = B(static_cast<Component>(transverse[1]));

        auto [Bt0L, Bt0R] = Reconstruction::template center_reconstruct<direction>(
            Bt0, index, projection<transverse[0]>());
        auto [Bt1L, Bt1R] = Reconstruction::template center_reconstruct<direction>(
            Bt1, index, projection<transverse[1]>());

        PerIndexVector<typename VecField::value_type> BL, BR;
        BL(direction)     = Bn(index);
        BR(direction)     = Bn(index);
        BL(transverse[0]) = Bt0L;
        BR(transverse[0]) = Bt0R;
        BL(transverse[1]) = Bt1L;
        BR(transverse[1]) = Bt1R;

        return std::make_pair(BL, BR);
    }

    template<auto direction, typename State, typename VecField>
    static auto transverse_reconstruct(State const& state, VecField const& B,
                                       MeshIndex<VecField::dimension> index)
    {
        auto constexpr transverse = []() {
            if constexpr (direction == Direction::X)
                return std::array{Direction::Y, Direction::Z};
            else if constexpr (direction == Direction::Y)
                return std::array{Direction::X, Direction::Z};
            else if constexpr (direction == Direction::Z)
                return std::array{Direction::X, Direction::Y};
        }();

        auto const Bn  = B(static_cast<Component>(direction));
        auto const Bt0 = B(static_cast<Component>(transverse[0]));
        auto const Bt1 = B(static_cast<Component>(transverse[1]));

        auto [Bt0L, Bt0R] = center_reconstruct_with_fallback_<direction>(
            state, Bt0, index, projection<transverse[0]>());
        auto [Bt1L, Bt1R] = center_reconstruct_with_fallback_<direction>(
            state, Bt1, index, projection<transverse[1]>());

        PerIndexVector<typename VecField::value_type> BL, BR;
        BL(direction)     = Bn(index);
        BR(direction)     = Bn(index);
        BL(transverse[0]) = Bt0L;
        BR(transverse[0]) = Bt0R;
        BL(transverse[1]) = Bt1L;
        BR(transverse[1]) = Bt1R;

        return std::make_pair(BL, BR);
    }

    template<auto direction, typename VecField>
    static auto reconstructed_laplacian(auto inverseMeshSize, VecField const& J,
                                        MeshIndex<VecField::dimension> index)
    {
        auto const& Jx = J(Component::X);
        auto const& Jy = J(Component::Y);
        auto const& Jz = J(Component::Z);

        auto const& [laplJxL, laplJxR] = reconstructed_laplacian_component_<direction>(
            inverseMeshSize, Jx, index, GridLayout::edgeXToCellCenter());

        auto const& [laplJyL, laplJyR] = reconstructed_laplacian_component_<direction>(
            inverseMeshSize, Jy, index, GridLayout::edgeYToCellCenter());

        auto const& [laplJzL, laplJzR] = reconstructed_laplacian_component_<direction>(
            inverseMeshSize, Jz, index, GridLayout::edgeZToCellCenter());

        return std::make_tuple(PerIndexVector{laplJxL, laplJyL, laplJzL},
                               PerIndexVector{laplJxR, laplJyR, laplJzR});
    }

private:
    template<typename T, typename = void>
    struct has_troubled_ : std::false_type
    {
    };

    template<typename T>
    struct has_troubled_<T, std::void_t<decltype(std::declval<T const&>().troubled)>>
        : std::true_type
    {
    };

    template<auto direction, typename State, typename Field>
    static auto reconstruct_with_fallback_(State const& S, Field const& F,
                                           MeshIndex<Field::dimension> index)
    {
        if constexpr (has_troubled_<State>::value)
        {
            if (S.troubled(index) > 0.0)
                return low_order_rec_t::template reconstruct<direction>(F, index);
        }
        return Reconstruction::template reconstruct<direction>(F, index);
    }

    template<auto direction, typename State, typename Field>
    static auto center_reconstruct_with_fallback_(State const& S, Field const& U,
                                                  MeshIndex<Field::dimension> index, auto projection)
    {
        if constexpr (has_troubled_<State>::value)
        {
            if (S.troubled(index) > 0.0)
                return low_order_rec_t::template center_reconstruct<direction>(U, index, projection);
        }
        return Reconstruction::template center_reconstruct<direction>(U, index, projection);
    }

    using low_order_rec_t = WENO3ZReconstruction<GridLayout>;

    template<auto direction, typename Field>
    static auto reconstructed_laplacian_component_(auto inverseMeshSize, Field const& J,
                                                   MeshIndex<Field::dimension> index,
                                                   auto projection)
    {
        auto d2 = [&](auto dir, auto const& prevValue, auto const& Value, auto const& nextValue) {
            return (inverseMeshSize[dir]) * (inverseMeshSize[dir])
                   * (prevValue - 2.0 * Value + nextValue);
        };

        auto const [JL, JR]
            = Reconstruction::template center_reconstruct<direction>(J, index, projection);

        MeshIndex<Field::dimension> prevX = GridLayout::template previous<Direction::X>(index);
        MeshIndex<Field::dimension> nextX = GridLayout::template next<Direction::X>(index);

        auto const [JL_X_1, JR_X_1]
            = Reconstruction::template center_reconstruct<direction>(J, prevX, projection);
        auto const [JL_X1, JR_X1]
            = Reconstruction::template center_reconstruct<direction>(J, nextX, projection);

        std::uint32_t dirX = static_cast<std::uint32_t>(Direction::X);

        if constexpr (Field::dimension == 1)
        {
            auto const LaplJL = d2(dirX, JL_X_1, JL, JL_X1);
            auto const LaplJR = d2(dirX, JR_X_1, JR, JR_X1);

            return std::make_tuple(LaplJL, LaplJR);
        }
        else if constexpr (Field::dimension >= 2)
        {
            MeshIndex<Field::dimension> prevY = GridLayout::template previous<Direction::Y>(index);
            MeshIndex<Field::dimension> nextY = GridLayout::template next<Direction::Y>(index);

            auto const [JL_Y_1, JR_Y_1]
                = Reconstruction::template center_reconstruct<direction>(J, prevY, projection);
            auto const [JL_Y1, JR_Y1]
                = Reconstruction::template center_reconstruct<direction>(J, nextY, projection);

            std::uint32_t dirY = static_cast<std::uint32_t>(Direction::Y);

            if constexpr (Field::dimension == 2)
            {
                auto const LaplJL = d2(dirX, JL_X_1, JL, JL_X1) + d2(dirY, JL_Y_1, JL, JL_Y1);
                auto const LaplJR = d2(dirX, JR_X_1, JR, JR_X1) + d2(dirY, JR_Y_1, JR, JR_Y1);

                return std::make_tuple(LaplJL, LaplJR);
            }
            if constexpr (Field::dimension == 3)
            {
                MeshIndex<Field::dimension> prevZ
                    = GridLayout::template previous<Direction::Z>(index);
                MeshIndex<Field::dimension> nextZ = GridLayout::template next<Direction::Z>(index);

                auto const [JL_Z_1, JR_Z_1]
                    = Reconstruction::template center_reconstruct<direction>(J, prevZ, projection);
                auto const [JL_Z1, JR_Z1]
                    = Reconstruction::template center_reconstruct<direction>(J, nextZ, projection);

                std::uint32_t dirZ = static_cast<std::uint32_t>(Direction::Z);

                auto const LaplJL = d2(dirX, JL_X_1, JL, JL_X1) + d2(dirY, JL_Y_1, JL, JL_Y1)
                                    + d2(dirZ, JL_Z_1, JL, JL_Z1);
                auto const LaplJR = d2(dirX, JR_X_1, JR, JR_X1) + d2(dirY, JR_Y_1, JR, JR_Y1)
                                    + d2(dirZ, JR_Z_1, JR, JR_Z1);

                return std::make_tuple(LaplJL, LaplJR);
            }
        }
    }
};
} // namespace PHARE::core

#endif
