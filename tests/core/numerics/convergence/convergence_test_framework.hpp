/**
 * @file convergence_test_framework.hpp
 * 
 * Reusable framework for convergence testing of numerical operators.
 * Provides templated utilities for:
 * - Running convergence studies over multiple grid resolutions
 * - Computing errors (L2, Linf norms)
 * - Calculating and reporting convergence orders
 * - Standardized test output formatting
 */

#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/field/field.hpp"
#include "core/utilities/types.hpp"
#include "core/utilities/point/point.hpp"

namespace PHARE::test
{

/**
 * @brief Compute convergence order from two consecutive errors
 * 
 * @param err_coarse Error on coarse grid
 * @param err_fine Error on fine grid  
 * @param refinement_ratio Grid refinement ratio (default: 2)
 * @return Convergence order
 */
inline double convergenceOrder(double err_coarse, double err_fine, double refinement_ratio = 2.0)
{
    if (err_fine == 0.0 || err_coarse == 0.0)
        return 0.0;
    return std::log(err_coarse / err_fine) / std::log(refinement_ratio);
}

/**
 * @brief Results from a convergence study
 */
struct ConvergenceResult
{
    std::vector<std::size_t> grid_sizes;  ///< Grid sizes tested
    std::vector<double> errors;            ///< Errors at each resolution
    std::vector<double> orders;            ///< Convergence orders between consecutive resolutions
    
    double min_order() const
    {
        if (orders.empty()) return 0.0;
        return *std::min_element(orders.begin(), orders.end());
    }
    
    double max_order() const
    {
        if (orders.empty()) return 0.0;
        return *std::max_element(orders.begin(), orders.end());
    }
    
    double avg_order() const
    {
        if (orders.empty()) return 0.0;
        double sum = 0.0;
        for (auto o : orders) sum += o;
        return sum / orders.size();
    }
};

/**
 * @brief Run a convergence study and return results
 * 
 * @tparam TestFunc Function signature: double(std::size_t grid_size)
 * @param test_function Function that runs test at given grid size and returns error
 * @param grid_sizes Vector of grid sizes to test
 * @param refinement_ratio Refinement ratio between consecutive grids
 * @return ConvergenceResult with errors and convergence orders
 */
template<typename TestFunc>
ConvergenceResult runConvergenceStudy(TestFunc&& test_function,
                                      std::vector<std::size_t> const& grid_sizes,
                                      double refinement_ratio = 2.0)
{
    ConvergenceResult result;
    result.grid_sizes = grid_sizes;
    result.errors.reserve(grid_sizes.size());
    
    // Run test at each resolution
    for (auto n : grid_sizes)
    {
        double error = test_function(n);
        result.errors.push_back(error);
    }
    
    // Compute convergence orders
    result.orders.reserve(grid_sizes.size() - 1);
    for (std::size_t i = 1; i < result.errors.size(); ++i)
    {
        double order = convergenceOrder(result.errors[i-1], result.errors[i], refinement_ratio);
        result.orders.push_back(order);
    }
    
    return result;
}

/**
 * @brief Print convergence results in a formatted table
 */
inline void printConvergenceTable(std::string const& test_name,
                                  ConvergenceResult const& result,
                                  std::ostream& out = std::cout)
{
    out << "\n=== " << test_name << " ===" << std::endl;
    out << std::setw(15) << "Grid Size" 
        << std::setw(20) << "Error" 
        << std::setw(15) << "Order" << std::endl;
    out << std::string(50, '-') << std::endl;
    
    for (std::size_t i = 0; i < result.grid_sizes.size(); ++i)
    {
        out << std::setw(15) << result.grid_sizes[i]
            << std::setw(20) << std::scientific << std::setprecision(6) << result.errors[i];
        
        if (i > 0)
        {
            out << std::setw(15) << std::fixed << std::setprecision(2) << result.orders[i-1];
        }
        out << std::endl;
    }
    
    if (!result.orders.empty())
    {
        out << std::string(50, '-') << std::endl;
        out << "Min order: " << std::fixed << std::setprecision(2) << result.min_order()
            << " | Max order: " << result.max_order()
            << " | Avg order: " << result.avg_order() << std::endl;
    }
}

/**
 * @brief Multi-quantity convergence study
 * 
 * Manages convergence testing for multiple quantities simultaneously.
 * Useful for testing multiple flux components, field components, etc.
 */
class MultiQuantityConvergenceStudy
{
public:
    /**
     * @brief Add a quantity to track
     */
    void addQuantity(std::string const& name)
    {
        results_[name] = ConvergenceResult{};
    }
    
    /**
     * @brief Record error for a quantity at current grid size
     */
    void recordError(std::string const& name, std::size_t grid_size, double error)
    {
        auto& result = results_[name];
        
        // Add grid size if not already present
        if (result.grid_sizes.empty() || result.grid_sizes.back() != grid_size)
        {
            result.grid_sizes.push_back(grid_size);
        }
        
        result.errors.push_back(error);
    }
    
    /**
     * @brief Compute convergence orders for all quantities
     */
    void computeOrders(double refinement_ratio = 2.0)
    {
        for (auto& [name, result] : results_)
        {
            result.orders.clear();
            for (std::size_t i = 1; i < result.errors.size(); ++i)
            {
                double order = convergenceOrder(result.errors[i-1], result.errors[i], refinement_ratio);
                result.orders.push_back(order);
            }
        }
    }
    
    /**
     * @brief Print summary table showing all quantities
     */
    void printSummary(std::ostream& out = std::cout) const
    {
        out << "\n=== Multi-Quantity Convergence Summary ===" << std::endl;
        out << std::setw(20) << "Quantity";
        
        // Print header with orders between consecutive resolutions
        if (!results_.empty() && !results_.begin()->second.orders.empty())
        {
            auto const& first_result = results_.begin()->second;
            for (std::size_t i = 0; i < first_result.orders.size(); ++i)
            {
                out << std::setw(12) << (std::to_string(first_result.grid_sizes[i]) + "→" 
                                        + std::to_string(first_result.grid_sizes[i+1]));
            }
        }
        out << std::endl;
        out << std::string(80, '-') << std::endl;
        
        // Print each quantity
        for (auto const& [name, result] : results_)
        {
            out << std::setw(20) << name;
            for (auto order : result.orders)
            {
                out << std::setw(12) << std::fixed << std::setprecision(2) << order;
            }
            out << std::endl;
        }
    }
    
    /**
     * @brief Check if all quantities meet minimum convergence order
     */
    bool allPass(double min_order) const
    {
        for (auto const& [name, result] : results_)
        {
            if (result.min_order() < min_order)
                return false;
        }
        return true;
    }
    
    /**
     * @brief Get list of failing quantities
     */
    std::vector<std::string> getFailures(double min_order) const
    {
        std::vector<std::string> failures;
        for (auto const& [name, result] : results_)
        {
            if (result.min_order() < min_order)
                failures.push_back(name);
        }
        return failures;
    }
    
    /**
     * @brief Get result for specific quantity
     */
    ConvergenceResult const& getResult(std::string const& name) const
    {
        return results_.at(name);
    }
    
private:
    std::map<std::string, ConvergenceResult> results_;
};

/**
 * @brief Error norms for field comparisons
 */
struct ErrorNorms
{
    double l1{0.0};    ///< L1 norm (average absolute error)
    double l2{0.0};    ///< L2 norm (RMS error)
    double linf{0.0};  ///< L-infinity norm (maximum absolute error)
    std::size_t count{0};  ///< Number of points used
};

/**
 * @brief Compute error norms between computed and exact fields
 * 
 * @tparam Field Field type with operator()(i,j,k) access
 * @tparam Layout Grid layout type
 * @tparam ExactFunc Function(x,y,z) -> exact value
 */
template<typename Layout, typename Field, typename ExactFunc>
ErrorNorms computeFieldError(Layout const& layout, 
                             Field const& field,
                             ExactFunc&& exact_fn,
                             int margin = 6)
{
    auto cent = layout.centering(field.physicalQuantity());
    auto psiX = layout.physicalStartIndex(cent[0], PHARE::core::Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], PHARE::core::Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], PHARE::core::Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], PHARE::core::Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], PHARE::core::Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], PHARE::core::Direction::Z);
    
    ErrorNorms norms;
    
    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));
                
                double computed = field(i, j, kk);
                double exact = exact_fn(c[0], c[1], c[2]);
                double diff = std::abs(computed - exact);
                
                norms.l1 += diff;
                norms.l2 += diff * diff;
                norms.linf = std::max(norms.linf, diff);
                norms.count++;
            }
    
    if (norms.count > 0)
    {
        norms.l1 /= static_cast<double>(norms.count);
        norms.l2 = std::sqrt(norms.l2 / static_cast<double>(norms.count));
    }
    
    return norms;
}

/**
 * @brief Configuration for convergence tests
 */
struct ConvergenceTestConfig
{
    std::vector<std::size_t> grid_sizes{16, 32, 64};  ///< Grid sizes to test
    double refinement_ratio{2.0};                      ///< Refinement ratio
    int margin{6};                                     ///< Margin to exclude from error calculation
    double min_order{1.75};                            ///< Minimum acceptable convergence order
    bool print_details{true};                          ///< Print detailed results
    
    /**
     * @brief Standard configuration for quick tests
     */
    static ConvergenceTestConfig quick()
    {
        return ConvergenceTestConfig{
            .grid_sizes = {16, 32},
            .margin = 4,
            .min_order = 1.5,
            .print_details = false
        };
    }
    
    /**
     * @brief Standard configuration for full tests
     */
    static ConvergenceTestConfig full()
    {
        return ConvergenceTestConfig{
            .grid_sizes = {16, 32, 64, 128},
            .margin = 6,
            .min_order = 3.5,
            .print_details = true
        };
    }
};

} // namespace PHARE::test
