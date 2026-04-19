#ifndef PHARE_TESTS_CORE_DATA_GRIDLAYOUT_GRIDLAYOUT_DERIV_HPP
#define PHARE_TESTS_CORE_DATA_GRIDLAYOUT_GRIDLAYOUT_DERIV_HPP


#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_impl.hpp"
#include "gridlayout_base_params.hpp"
#include "gridlayout_params.hpp"
#include "gridlayout_utilities.hpp"
#include "core/physical_quantities.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/point/point.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace PHARE::core;

std::vector<double> read(std::string filename);



template<typename GridLayoutImpl>
class a1DDerivative : public ::testing::Test
{
public:
    using GridLayout_t = GridLayout<GridLayoutImpl>;

    a1DDerivative()
        : layout{{{0.1}}, {50}, Point{0.}}
        , By{"By", PhysicalQuantity::Scalar::By, layout.allocSize(PhysicalQuantity::Scalar::By)}
        , Ez{"Ez", PhysicalQuantity::Scalar::Ez, layout.allocSize(PhysicalQuantity::Scalar::Ez)}
    {
    }

protected:
    GridLayout_t layout;
    static constexpr std::size_t interp_order = GridLayoutImpl::interp_order;
    Grid<NdArrayVector<1>, PhysicalQuantity::Scalar> By;
    Grid<NdArrayVector<1>, PhysicalQuantity::Scalar> Ez;
};



template<typename GridLayoutImpl>
class a2DDerivative : public ::testing::Test
{
public:
    using GridLayout_t = GridLayout<GridLayoutImpl>;

    a2DDerivative()
        : layout{{{0.1, 0.2}}, {50, 30}, Point{0., 0.}}
        , By{"By", PhysicalQuantity::Scalar::By, layout.allocSize(PhysicalQuantity::Scalar::By)}
        , Ez{"Ez", PhysicalQuantity::Scalar::Ez, layout.allocSize(PhysicalQuantity::Scalar::Ez)}
    {
    }

protected:
    GridLayout_t layout;
    static constexpr std::size_t interp_order = GridLayoutImpl::interp_order;
    Grid<NdArrayVector<2>, PhysicalQuantity::Scalar> By;
    Grid<NdArrayVector<2>, PhysicalQuantity::Scalar> Ez;
};



template<typename GridLayoutImpl>
class a3DDerivative : public ::testing::Test
{
public:
    using GridLayout_t = GridLayout<GridLayoutImpl>;
    a3DDerivative()
        : layout{{{0.1, 0.2, 0.3}}, {50, 30, 40}, Point{0., 0., 0.}}
        , By{"By", PhysicalQuantity::Scalar::By, layout.allocSize(PhysicalQuantity::Scalar::By)}
        , Ez{"Ez", PhysicalQuantity::Scalar::Ez, layout.allocSize(PhysicalQuantity::Scalar::Ez)}
    {
    }

protected:
    GridLayout_t layout;
    static constexpr std::size_t interp_order = GridLayoutImpl::interp_order;
    Grid<NdArrayVector<3>, PhysicalQuantity::Scalar> By;
    Grid<NdArrayVector<3>, PhysicalQuantity::Scalar> Ez;
};


#endif
