#include <amr/physical_models/physical_model.hpp>
#include <core/data/vecfield/vecfield.hpp>
#include <core/data/vecfield/vecfield_component.hpp>
#include <core/numerics/godunov_fluxes/godunov_fluxes.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "amr/types/amr_types.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "amr/messengers/messenger.hpp"


#include "amr/wrappers/hierarchy.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/numerics/boundary_condition/boundary_condition.hpp"
#include "phare_core.hpp"
#include "tests/core/data/field/test_field_fixtures_mhd.hpp"
#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "amr/solvers/solver_mhd.hpp"
#include "amr/solvers/solver.hpp"

#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

static constexpr std::size_t dim = 1, order = 1;
constexpr std::uint32_t cellnbr = 65;

using YeeLayout_t       = PHARE::core::GridLayoutImplYeeMHD<dim, order>;
using GridLayoutMHD     = PHARE::core::GridLayout<YeeLayout_t>;
using GridLayout_t      = TestGridLayout<GridLayoutMHD>;
using BoundaryCondition = PHARE::core::BoundaryCondition<dim, order>;
using MHDQuantity       = PHARE::core::MHDQuantity;
using FieldMHD          = PHARE::core::FieldMHD<dim>;
using VecFieldMHD       = PHARE::core::VecField<FieldMHD, MHDQuantity>;
using UsableMHDState_t  = PHARE::core::UsableMHDState<dim>;

using namespace PHARE::initializer;

PHAREDict getDict()
{
    PHAREDict dict;

    dict["godunov"]["resistivity"]                 = 0.01;
    dict["godunov"]["hyper_resistivity"]           = 0.01;
    dict["godunov"]["heat_capacity_ratio"]         = 0.01;
    dict["to_primitive"]["heat_capacity_ratio"]    = 0.01;
    dict["to_conservative"]["heat_capacity_ratio"] = 0.01;

    return dict;
}

struct DummyModelViewConstructor
{
    using GodunovFluxes_t = PHARE::core::GodunovFluxes<GridLayout_t>;

    DummyModelViewConstructor(GridLayout_t const& layout)
        : rho_x{"rho_x", layout, MHDQuantity::Scalar::ScalarFlux_x}
        , rhoV_x{"rhoV_x", layout, MHDQuantity::Vector::VecFlux_x}
        , B_x{"B_x", layout, MHDQuantity::Vector::VecFlux_x}
        , Etot_x{"Etot_x", layout, MHDQuantity::Scalar::ScalarFlux_x}

        , rho_y{"rho_y", layout, MHDQuantity::Scalar::ScalarFlux_y}
        , rhoV_y{"rhoV_y", layout, MHDQuantity::Vector::VecFlux_y}
        , B_y{"B_y", layout, MHDQuantity::Vector::VecFlux_y}
        , Etot_y{"Etot_y", layout, MHDQuantity::Scalar::ScalarFlux_y}

        , rho_z{"rho_z", layout, MHDQuantity::Scalar::ScalarFlux_z}
        , rhoV_z{"rhoV_z", layout, MHDQuantity::Vector::VecFlux_z}
        , B_z{"B_z", layout, MHDQuantity::Vector::VecFlux_z}
        , Etot_z{"Etot_z", layout, MHDQuantity::Scalar::ScalarFlux_z}

        , layouts{layout}
    {
    }

    PHARE::core::UsableFieldMHD<dim> rho_x;
    PHARE::core::UsableVecFieldMHD<dim> rhoV_x;
    PHARE::core::UsableVecFieldMHD<dim> B_x;
    PHARE::core::UsableFieldMHD<dim> Etot_x;

    PHARE::core::UsableFieldMHD<dim> rho_y;
    PHARE::core::UsableVecFieldMHD<dim> rhoV_y;
    PHARE::core::UsableVecFieldMHD<dim> B_y;
    PHARE::core::UsableFieldMHD<dim> Etot_y;

    PHARE::core::UsableFieldMHD<dim> rho_z;
    PHARE::core::UsableVecFieldMHD<dim> rhoV_z;
    PHARE::core::UsableVecFieldMHD<dim> B_z;
    PHARE::core::UsableFieldMHD<dim> Etot_z;

    GridLayout_t layouts;
};


struct DummyHierarchy
{
    auto getPatchLevel(std::size_t lvl) const
    {
        int* a = nullptr;
        *a     = 1;
        return a;
    };
};


struct DummyTypes
{
    using patch_t     = PHARE::amr::SAMRAI_Types::patch_t;
    using level_t     = int;
    using hierarchy_t = DummyHierarchy;
};


struct DummyRessourcesManager
{
};


struct DummyMHDModel : public PHARE::solver::IPhysicalModel<DummyTypes>
{
    using field_type                       = FieldMHD;
    using vecfield_type                    = VecFieldMHD;
    using gridlayout_type                  = GridLayout_t;
    static constexpr std::size_t dimension = dim;
    static constexpr auto model_name       = "mhd_model";

    DummyMHDModel(GridLayout_t layout)
        : PHARE::solver::IPhysicalModel<DummyTypes>(model_name)
        , usablestate{layout}
        , state{usablestate.super()}
    {
    }

    void initialize(level_t& level) override {}

    void allocate(patch_t& patch, double const allocateTime) override {}

    void fillMessengerInfo(std::unique_ptr<PHARE::amr::IMessengerInfo> const& info) const override
    {
    }

    PHARE::core::UsableMHDState<dimension> usablestate;
    PHARE::core::MHDState<VecFieldMHD>& state;
};

struct DummyModelView : public PHARE::solver::ISolverModelView
{
    using GodunovFluxes_t = PHARE::core::GodunovFluxes<GridLayout_t>;

    DummyModelView(DummyModelViewConstructor& construct, DummyMHDModel& _model)
        : model_{_model}
    {
        rho.push_back(&model_.state.rho);
        V.push_back(&model_.state.V);
        B.push_back(&model_.state.B);
        P.push_back(&model_.state.P);
        J.push_back(&model_.state.J);
        E.push_back(&model_.state.E);

        rhoV.push_back(&model_.state.rhoV);
        Etot.push_back(&model_.state.Etot);

        rho_x.push_back(&construct.rho_x.super());
        rhoV_x.push_back(&construct.rhoV_x.super());
        B_x.push_back(&construct.B_x.super());
        Etot_x.push_back(&construct.Etot_x.super());

        rho_y.push_back(&construct.rho_y.super());
        rhoV_y.push_back(&construct.rhoV_y.super());
        B_y.push_back(&construct.B_y.super());
        Etot_y.push_back(&construct.Etot_y.super());

        rho_z.push_back(&construct.rho_z.super());
        rhoV_z.push_back(&construct.rhoV_z.super());
        B_z.push_back(&construct.B_z.super());
        Etot_z.push_back(&construct.Etot_z.super());

        layouts.push_back(&construct.layouts);
    }

    auto& model() { return model_; }
    auto& model() const { return model_; }

    std::vector<FieldMHD*> rho;
    std::vector<VecFieldMHD*> V;
    std::vector<VecFieldMHD*> B;
    std::vector<FieldMHD*> P;
    std::vector<VecFieldMHD*> J;
    std::vector<VecFieldMHD*> E;

    std::vector<VecFieldMHD*> rhoV;
    std::vector<FieldMHD*> Etot;

    std::vector<FieldMHD*> rho_x;
    std::vector<VecFieldMHD*> rhoV_x;
    std::vector<VecFieldMHD*> B_x;
    std::vector<FieldMHD*> Etot_x;

    std::vector<FieldMHD*> rho_y;
    std::vector<VecFieldMHD*> rhoV_y;
    std::vector<VecFieldMHD*> B_y;
    std::vector<FieldMHD*> Etot_y;

    std::vector<FieldMHD*> rho_z;
    std::vector<VecFieldMHD*> rhoV_z;
    std::vector<VecFieldMHD*> B_z;
    std::vector<FieldMHD*> Etot_z;

    std::vector<GridLayout_t*> layouts;

    DummyMHDModel& model_;
};

class DummyMessenger : public PHARE::amr::IMessenger<PHARE::solver::IPhysicalModel<DummyTypes>>
{
    using IPhysicalModel = PHARE::solver::IPhysicalModel<DummyTypes>;
    using level_t        = DummyTypes::level_t;

    std::string name() override { return "DummyMessenger"; }

    std::unique_ptr<PHARE::amr::IMessengerInfo> emptyInfoFromCoarser() override
    {
        return std::make_unique<PHARE::amr::IMessengerInfo>();
    }

    std::unique_ptr<PHARE::amr::IMessengerInfo> emptyInfoFromFiner() override
    {
        return std::make_unique<PHARE::amr::IMessengerInfo>();
    }

    void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override {}

    void registerQuantities(std::unique_ptr<PHARE::amr::IMessengerInfo> fromCoarserInfo,
                            std::unique_ptr<PHARE::amr::IMessengerInfo> fromFinerInfo) override
    {
    }

    void registerLevel(const std::shared_ptr<SAMRAI::hier::PatchHierarchy>& hierarchy,
                       int level) override
    {
    }

    void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                int const levelNumber, std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                IPhysicalModel& model, double const initDataTime) override
    {
    }

    void initLevel(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                   double const initDataTime) override
    {
    }

    void firstStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                   const std::shared_ptr<SAMRAI::hier::PatchHierarchy>& hierarchy,
                   double const currentTime, double const prevCoarserTime,
                   double const newCoarserTime) override
    {
    }

    void lastStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level) override {}

    void prepareStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                     double currentTime) override
    {
    }

    void fillRootGhosts(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                        double const initDataTime) override
    {
    }

    void synchronize(SAMRAI::hier::PatchLevel& level) override {}

    void postSynchronize(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                         double const time) override
    {
    }

    std::string fineModelName() const override { return "mhd_model"; }
    std::string coarseModelName() const override { return "mhd_model"; }

    template<typename Field>
    void fillGhost_(Field& F, auto nx, auto ny, auto nz)
    {
        static constexpr auto dimension = Field::dimension;
        static constexpr auto nghost    = order;

        if constexpr (dimension == 1)
        {
            for (auto g = 1u; g <= nghost; ++g)
            {
                F(nghost - g)          = F(nx - g - nghost); // left ghost
                F(nx - 1 + g - nghost) = F(g - 1 + nghost);  // right ghost
            }
        }
        else if constexpr (dimension == 2)
        {
            for (auto g = 1u; g <= nghost; ++g)
            {
                for (auto i = nghost; i < nx - nghost; ++i)
                {
                    F(i, nghost - g)          = F(i, ny - g - nghost); // bottom ghost
                    F(i, ny - 1 + g - nghost) = F(i, g - 1 + nghost);  // top ghost
                }
                for (auto j = nghost; j < ny - nghost; ++j)
                {
                    F(nghost - g, j)          = F(nx - g - nghost, j); // left ghost
                    F(nx - 1 + g - nghost, j) = F(g - 1 + nghost, j);  // right ghost
                }
            }
            // corners
            for (auto g1 = 1u; g1 <= nghost; ++g1)
            {
                for (auto g2 = 1u; g2 <= nghost; ++g2)
                {
                    F(nghost - g1, nghost - g2)
                        = F(nx - g1 - nghost, ny - g2 - nghost); // bottom left
                    F(nx - 1 + g1 - nghost, nghost - g2)
                        = F(g1 - 1 + nghost, ny - g2 - nghost); // bottom right
                    F(nghost - g1, ny - 1 + g2 - nghost)
                        = F(nx - g1 - nghost, g2 - 1 + nghost); // top left
                    F(nx - 1 + g1 - nghost, ny - 1 + g2 - nghost)
                        = F(g1 - 1 + nghost, g2 - 1 + nghost); // top right
                }
            }
        }
        else if constexpr (dimension == 3)
        {
            for (auto g = 1u; g <= nghost; ++g)
            {
                for (auto i = nghost; i < nx - nghost; ++i)
                {
                    for (auto j = nghost; j < ny - nghost; ++j)
                    {
                        F(i, j, nghost - g)          = F(i, j, nz - g - nghost); // bottom ghost
                        F(i, j, nz - 1 + g - nghost) = F(i, j, g - 1 + nghost);  // top ghost
                    }
                    for (auto k = nghost; k < nz - nghost; ++k)
                    {
                        F(i, nghost - g, k)          = F(i, ny - g - nghost, k); // left ghost
                        F(i, ny - 1 + g - nghost, k) = F(i, g - 1 + nghost, k);  // right ghost
                    }
                }
                for (auto j = nghost; j < ny - nghost; ++j)
                {
                    for (auto k = nghost; k < nz - nghost; ++k)
                    {
                        F(nghost - g, j, k)          = F(nx - g - nghost, j, k); // front ghost
                        F(nx - 1 + g - nghost, j, k) = F(g - 1 + nghost, j, k);  // back ghost
                    }
                }
            }
            // corners
            for (auto g1 = 1u; g1 <= nghost; ++g1)
            {
                for (auto g2 = 1u; g2 <= nghost; ++g2)
                {
                    for (auto g3 = 1u; g3 <= nghost; ++g3)
                    {
                        F(nghost - g1, nghost - g2, nghost - g3)
                            = F(nx - g1 - nghost, ny - g2 - nghost,
                                nz - g3 - nghost); // bottom front left
                        F(nx - 1 + g1 - nghost, nghost - g2, nghost - g3)
                            = F(g1 - 1 + nghost, ny - g2 - nghost,
                                nz - g3 - nghost); // bottom front right
                        F(nghost - g1, ny - 1 + g2 - nghost, nghost - g3)
                            = F(nx - g1 - nghost, g2 - 1 + nghost,
                                nz - g3 - nghost); // bottom back left
                        F(nx - 1 + g1 - nghost, ny - 1 + g2 - nghost, nghost - g3)
                            = F(g1 - 1 + nghost, g2 - 1 + nghost,
                                nz - g3 - nghost); // bottom back right
                        F(nghost - g1, nghost - g2, nz - 1 + g3 - nghost)
                            = F(nx - g1 - nghost, ny - g2 - nghost,
                                g3 - 1 + nghost); // top front left
                        F(nx - 1 + g1 - nghost, nghost - g2, nz - 1 + g3 - nghost)
                            = F(g1 - 1 + nghost, ny - g2 - nghost,
                                g3 - 1 + nghost); // top front right
                    }
                }
            }
        }
    }

public:
    template<typename Field>
    void fillMomentGhost(Field& F, level_t& level, double const newTime)
    {
        fillGhost_(F, cellnbr, cellnbr, cellnbr);
    };

    template<typename VecField>
    void fillMagneticGhost(VecField& B, level_t& level, double const newTime)
    {
        auto& Bx = B(PHARE::core::Component::X);
        auto& By = B(PHARE::core::Component::Y);
        auto& Bz = B(PHARE::core::Component::Z);

        fillGhost_(Bx, cellnbr + 1, cellnbr, cellnbr);
        fillGhost_(By, cellnbr, cellnbr + 1, cellnbr);
        fillGhost_(Bz, cellnbr, cellnbr, cellnbr + 1);
    }

    template<typename VecField>
    void fillCurrentGhost(VecField& J, level_t& level, double const newTime)
    {
        auto& Jx = J(PHARE::core::Component::X);
        auto& Jy = J(PHARE::core::Component::Y);
        auto& Jz = J(PHARE::core::Component::Z);

        fillGhost_(Jx, cellnbr, cellnbr + 1, cellnbr + 1);
        fillGhost_(Jy, cellnbr + 1, cellnbr, cellnbr + 1);
        fillGhost_(Jz, cellnbr + 1, cellnbr + 1, cellnbr);
    }

    template<typename VecField>
    void fillElectricGhost(VecField& E, level_t& level, double const newTime)
    {
        auto& Ex = E(PHARE::core::Component::X);
        auto& Ey = E(PHARE::core::Component::Y);
        auto& Ez = E(PHARE::core::Component::Z);

        fillGhost_(Ex, cellnbr, cellnbr + 1, cellnbr + 1);
        fillGhost_(Ey, cellnbr + 1, cellnbr, cellnbr + 1);
        fillGhost_(Ez, cellnbr + 1, cellnbr + 1, cellnbr);
    }
};


TEST(MHDSolverTest, Test)
{
    GridLayout_t layout{cellnbr};
    DummyMHDModel model(layout);
    DummyModelViewConstructor dummy_view_construct(layout);
    DummyModelView dummy_view(dummy_view_construct, model);
    PHARE::solver::SolverMHD<DummyMHDModel, DummyTypes, DummyMessenger, DummyModelView>
        TestMHDSolver(getDict());


    DummyMessenger dummy_messenger;

    DummyHierarchy dummy_hierachy;

    TestMHDSolver.advanceLevel(dummy_hierachy, 1, dummy_view, dummy_messenger, 0.0, 0.01);
    ASSERT_NO_THROW();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
