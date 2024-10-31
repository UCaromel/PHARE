#include <amr/physical_models/physical_model.hpp>
#include <core/data/vecfield/vecfield.hpp>
#include <core/numerics/godunov_fluxes/godunov_fluxes.hpp>
#include <cstddef>
#include <gtest/gtest.h>
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

constexpr std::uint32_t cells = 65;
constexpr std::size_t dim = 1, interp = 1;

using YeeLayout_t       = PHARE::core::GridLayoutImplYeeMHD<dim, interp>;
using GridLayoutMHD     = PHARE::core::GridLayout<YeeLayout_t>;
using GridLayout_t      = TestGridLayout<GridLayoutMHD>;
using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;
using MHDQuantity       = PHARE::core::MHDQuantity;
using FieldMHD          = PHARE::core::FieldMHD<dim>;
using VecFieldMHD       = PHARE::core::VecField<FieldMHD, MHDQuantity>;

using namespace PHARE::initializer;

PHAREDict getDict()
{
    PHAREDict dict;

    dict["godunov"]["resistivity"]         = 0.01;
    dict["godunov"]["hyper_resistivity"]   = 0.01;
    dict["godunov"]["heat_capacity_ratio"] = 0.01;

    return dict;
}

struct DummyModelViewConstructor : public PHARE::solver::ISolverModelView
{
    using GodunovFluxes_t = PHARE::core::GodunovFluxes<GridLayout_t>;

    DummyModelViewConstructor(GridLayout_t const& layout)
        : rho{"rho", layout, MHDQuantity::Scalar::rho}
        , V{"V", layout, MHDQuantity::Vector::V}
        , B_CT{"B_CT", layout, MHDQuantity::Vector::B_CT}
        , P{"P", layout, MHDQuantity::Scalar::P}
        , J{"J", layout, MHDQuantity::Vector::J}

        , rho_x{"rho_x", layout, MHDQuantity::Scalar::ScalarFlux_x}
        , rhoV_x{"V_x", layout, MHDQuantity::Vector::VecFlux_x}
        , B_x{"B_x", layout, MHDQuantity::Vector::VecFlux_x}
        , Etot_x{"Etot_x", layout, MHDQuantity::Scalar::ScalarFlux_x}

        , rho_y{"rho_y", layout, MHDQuantity::Scalar::ScalarFlux_y}
        , rhoV_y{"V_y", layout, MHDQuantity::Vector::VecFlux_y}
        , B_y{"B_y", layout, MHDQuantity::Vector::VecFlux_y}
        , Etot_y{"Etot_y", layout, MHDQuantity::Scalar::ScalarFlux_y}

        , rho_z{"rho_z", layout, MHDQuantity::Scalar::ScalarFlux_z}
        , rhoV_z{"V_z", layout, MHDQuantity::Vector::VecFlux_z}
        , B_z{"B_z", layout, MHDQuantity::Vector::VecFlux_z}
        , Etot_z{"Etot_z", layout, MHDQuantity::Scalar::ScalarFlux_z}

        , layouts{layout}
    {
    }

    PHARE::core::UsableFieldMHD<dim> rho;
    PHARE::core::UsableVecFieldMHD<dim> V;
    PHARE::core::UsableVecFieldMHD<dim> B_CT;
    PHARE::core::UsableFieldMHD<dim> P;
    PHARE::core::UsableVecFieldMHD<dim> J;

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

struct DummyModelView : public PHARE::solver::ISolverModelView
{
    using GodunovFluxes_t = PHARE::core::GodunovFluxes<GridLayout_t>;

    DummyModelView(DummyModelViewConstructor& construct)
    {
        rho.push_back(&construct.rho.super());
        V.push_back(&construct.V.super());
        B_CT.push_back(&construct.B_CT.super());
        P.push_back(&construct.P.super());
        J.push_back(&construct.J.super());

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

    std::vector<FieldMHD*> rho;
    std::vector<VecFieldMHD*> V;
    std::vector<VecFieldMHD*> B_CT;
    std::vector<FieldMHD*> P;
    std::vector<VecFieldMHD*> J;

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
};


class DummyMessenger : public PHARE::amr::IMessenger<PHARE::solver::IPhysicalModel<DummyTypes>>
{
    using IPhysicalModel = PHARE::solver::IPhysicalModel<DummyTypes>;

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

    std::string fineModelName() const override { return "DummyMHDModel"; }
    std::string coarseModelName() const override { return "DummyMHDModel"; }
};


TEST(UsableMHDStateTest, ConstructionTest)
{
    GridLayout_t layout{cells};
    DummyModelViewConstructor dummy_view_construct(layout);
    DummyModelView dummy_view(dummy_view_construct);
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
