#ifndef PHARE_CORE_DATA_NDARRAY_NDARRAY_VECTOR_HPP
#define PHARE_CORE_DATA_NDARRAY_NDARRAY_VECTOR_HPP

#include "core/def.hpp"
#include <stdexcept>
#include <array>
#include <cstdint>
#include <vector>
#include <tuple>
#include <numeric>
#include <iostream>


#include "core/utilities/types.hpp"


namespace PHARE::core
{
template<std::size_t dim, bool c_ordering = true, typename DataType = double>
struct NdArrayViewer
{
    template<typename NCells, typename... Indexes>
    NO_DISCARD static DataType const& at(DataType const* data, NCells const& nCells,
                                         Indexes const&... indexes)
    {
        auto params = std::forward_as_tuple(indexes...);
        static_assert(sizeof...(Indexes) == dim);
        // static_assert((... && std::is_unsigned_v<decltype(indexes)>)); TODO : manage later if
        // this test should be included

        if constexpr (dim == 1)
        {
            auto i = std::get<0>(params);

            return data[i];
        }

        if constexpr (dim == 2)
        {
            auto i = std::get<0>(params);
            auto j = std::get<1>(params);

            if constexpr (c_ordering)
                return data[j + i * nCells[1]];
            else
                return data[i + j * nCells[0]];
        }

        if constexpr (dim == 3)
        {
            auto i = std::get<0>(params);
            auto j = std::get<1>(params);
            auto k = std::get<2>(params);

            if constexpr (c_ordering)
                return data[k + j * nCells[2] + i * nCells[1] * nCells[2]];
            else
                return data[i + j * nCells[0] + k * nCells[1] * nCells[0]];
        }
    }

    template<typename NCells, template<typename, std::size_t> typename Indexes, typename Index>
    NO_DISCARD static DataType const& at(DataType const* data, NCells const& nCells,
                                         Indexes<Index, dim> const& indexes)

    {
        if constexpr (dim == 1)
            return at(data, nCells, indexes[0]);

        else if constexpr (dim == 2)
            return at(data, nCells, indexes[0], indexes[1]);

        else if constexpr (dim == 3)
            return at(data, nCells, indexes[0], indexes[1], indexes[2]);
    }
};



template<typename Array, typename Mask>
class MaskedView
{
public:
    static auto constexpr dimension = Array::dimension;
    using DataType                  = typename Array::type;
    using data_type                 = typename Array::type;

    MaskedView(Array& array, Mask const& mask)
        : array_{array}
        , shape_{array.shape()}
        , mask_{mask}
    {
    }

    MaskedView(Array& array, Mask&& mask)
        : array_{array}
        , shape_{array.shape()}
        , mask_{std::move(mask)}
    {
    }

    template<typename... Indexes>
    NO_DISCARD DataType const& operator()(Indexes... indexes) const
    {
        return NdArrayViewer<dimension, true, DataType>::at(array_.data(), shape_, indexes...);
    }

    template<typename... Indexes>
    NO_DISCARD DataType& operator()(Indexes... indexes)
    {
        return const_cast<DataType&>(static_cast<MaskedView const&>(*this)(indexes...));
    }

    NO_DISCARD auto operator=(data_type value) { mask_.fill(array_, value); }

    NO_DISCARD auto xstart() const { return mask_.min(); }

    NO_DISCARD auto xend() const { return shape_[0] - 1 - mask_.max(); }


    NO_DISCARD auto ystart() const { return mask_.min(); }

    NO_DISCARD auto yend() const { return shape_[1] - 1 - mask_.max(); }


private:
    Array& array_;
    std::array<std::uint32_t, dimension> shape_;
    Mask const& mask_;
};



template<std::size_t dim, typename DataType = double, bool c_ordering = true>
class NdArrayView
{
public:
    static constexpr bool is_contiguous = 1;
    static std::size_t const dimension  = dim;
    using type                          = DataType;
    using pointer_type                  = DataType*;
    using viewer                        = NdArrayViewer<dim, c_ordering, DataType>;

    explicit NdArrayView(pointer_type ptr, std::array<std::uint32_t, dim> const& nCells)
        : ptr_{ptr}
        , nCells_{nCells}
    {
    }

    explicit NdArrayView(std::vector<std::decay_t<DataType>> const& v,
                         std::array<std::uint32_t, dim> const& nbCell)
        : NdArrayView{v.data(), nbCell}
    {
    }

    template<typename... Indexes>
    NO_DISCARD DataType& operator()(Indexes... indexes)
    {
        return const_cast<DataType&>(static_cast<NdArrayView const&>(*this)(indexes...));
    }

    template<typename... Indexes>
    NO_DISCARD DataType const& operator()(Indexes... indexes) const
    {
        return viewer::at(ptr_, nCells_, indexes...);
    }

    template<typename Index>
    NO_DISCARD DataType const& operator()(std::array<Index, dim> const& indexes) const
    {
        return viewer::at(ptr_, nCells_, indexes);
    }

    template<typename Index>
    NO_DISCARD DataType& operator()(std::array<Index, dim> const& indexes)
    {
        return const_cast<DataType&>(static_cast<NdArrayView const&>(*this)(indexes));
    }

    NO_DISCARD auto data() const { return ptr_; }
    NO_DISCARD auto size() const
    {
        return std::accumulate(nCells_.begin(), nCells_.end(), std::size_t{1},
                               std::multiplies<std::size_t>());
    }
    NO_DISCARD auto shape() const { return nCells_; }

    void fill_from(NdArrayView const& that)
    {
        if (for_N_any<dim>([&](auto i) { return shape()[i] != that.shape()[i]; }))
            throw std::runtime_error("ArrayView::fill_from: Incompatible input shape");
        std::copy(that.data(), that.data() + size(), data());
    }

    NO_DISCARD auto begin() const { return ptr_; }
    NO_DISCARD auto begin() { return ptr_; }

    NO_DISCARD auto end() const { return ptr_ + size(); }
    NO_DISCARD auto end() { return ptr_ + size(); }

    void zero() { fill(0); }
    auto& fill(DataType const& v)
    {
        std::fill(begin(), end(), v);
        return *this;
    }

    void setBuffer(pointer_type ptr) { ptr_ = ptr; }
    void setShape(std::array<std::uint32_t, dim> const nCells) { nCells_ = nCells; }

private:
    pointer_type ptr_ = nullptr;
    std::array<std::uint32_t, dim> nCells_;
};


template<bool c_ordering = true, typename DataType, std::size_t dim>
auto make_array_view(DataType* data, std::array<std::uint32_t, dim> const& shape)
{
    return NdArrayView<dim, DataType, c_ordering>{data, shape};
}

template<bool c_ordering = true, typename DataType, std::size_t dim>
auto make_array_view(DataType const* const data, std::array<std::uint32_t, dim> shape)
{
    return NdArrayView<dim, DataType const, c_ordering>{data, shape};
}


template<std::size_t dim, typename DataType = double, bool c_ordering = true>
class NdArrayVector
{
public:
    static constexpr bool is_contiguous = 1;
    static std::size_t const dimension  = dim;
    using type                          = DataType;

    NdArrayVector() = delete;

    template<typename... Nodes>
    explicit NdArrayVector(Nodes... nodes)
        : nCells_{nodes...}
        , data_((... * nodes))
    {
        static_assert(sizeof...(Nodes) == dim);
    }

    explicit NdArrayVector(std::array<std::uint32_t, dim> const& ncells)
        : nCells_{ncells}
        , data_(std::accumulate(ncells.begin(), ncells.end(), 1, std::multiplies<int>()))
    {
    }

    NdArrayVector(NdArrayVector const& source)            = default;
    NdArrayVector(NdArrayVector&& source)                 = default;
    NdArrayVector& operator=(NdArrayVector const& source) = default;
    NdArrayVector& operator=(NdArrayVector&& source)      = default;

    NO_DISCARD auto data() const { return data_.data(); }
    NO_DISCARD auto data() { return data_.data(); }

    NO_DISCARD auto size() const { return data_.size(); }

    NO_DISCARD auto begin() const { return std::begin(data_); }
    NO_DISCARD auto begin() { return std::begin(data_); }

    NO_DISCARD auto end() const { return std::end(data_); }
    NO_DISCARD auto end() { return std::end(data_); }


    template<typename... Indexes>
    NO_DISCARD DataType const& operator()(Indexes... indexes) const
    {
        return NdArrayViewer<dim, c_ordering, DataType>::at(data_.data(), nCells_, indexes...);
    }

    template<typename... Indexes>
    NO_DISCARD DataType& operator()(Indexes... indexes)
    {
        return const_cast<DataType&>(static_cast<NdArrayVector const&>(*this)(indexes...));
    }

    template<typename Index>
    NO_DISCARD DataType const& operator()(std::array<Index, dim> const& indexes) const
    {
        return NdArrayViewer<dim, c_ordering, DataType>::at(data_.data(), nCells_, indexes);
    }

    template<typename Index>
    NO_DISCARD DataType& operator()(std::array<Index, dim> const& indexes)
    {
        return const_cast<DataType&>(static_cast<NdArrayVector const&>(*this)(indexes));
    }


    NO_DISCARD auto& shape() const { return nCells_; }

    template<typename Mask>
    NO_DISCARD auto operator[](Mask&& mask)
    {
        return MaskedView{*this, std::forward<Mask>(mask)};
    }


    NO_DISCARD auto& vector() { return data_; }
    NO_DISCARD auto& vector() const { return data_; }


private:
    std::array<std::uint32_t, dim> nCells_;
    std::vector<DataType> data_;
};


class NdArrayMask
{
public:
    NdArrayMask(std::size_t min, std::size_t max)
        : min_{min}
        , max_{max}
    {
    }

    NdArrayMask(std::size_t width)
        : min_{width}
        , max_{width}
    {
    }

    template<typename Array>
    void fill(Array& array, typename Array::type val) const
    {
        if constexpr (Array::dimension == 1)
            fill1D(array, val);

        else if constexpr (Array::dimension == 2)
            fill2D(array, val);

        else if constexpr (Array::dimension == 3)
            fill3D(array, val);
    }

    template<typename Array>
    void fill1D(Array& array, typename Array::type val) const
    {
        auto shape = array.shape();

        for (std::size_t i = min_; i <= max_; ++i)
            array(i) = val;

        for (std::size_t i = shape[0] - 1 - max_; i <= shape[0] - 1 - min_; ++i)
            array(i) = val;
    }

    template<typename Array>
    void fill2D(Array& array, typename Array::type val) const
    {
        auto shape = array.shape();

        // left border
        for (std::size_t i = min_; i <= max_; ++i)
            for (std::size_t j = min_; j <= shape[1] - 1 - max_; ++j)
                array(i, j) = val;

        // right border
        for (std::size_t i = shape[0] - 1 - max_; i <= shape[0] - 1 - min_; ++i)
            for (std::size_t j = min_; j <= shape[1] - 1 - max_; ++j)
                array(i, j) = val;


        for (std::size_t i = min_; i <= shape[0] - 1 - min_; ++i)
        {
            // bottom border
            for (std::size_t j = min_; j <= max_; ++j)
                array(i, j) = val;

            // top border
            for (std::size_t j = shape[1] - 1 - max_; j <= shape[1] - 1 - min_; ++j)
                array(i, j) = val;
        }
    }

    template<typename Array>
    void fill3D(Array& array, typename Array::type val) const
    {
        throw std::runtime_error("3d not implemented");
    }

    template<typename Array>
    NO_DISCARD auto nCells(Array const& array)
    {
        auto shape = array.shape();

        std::size_t cells = 0;

        if constexpr (Array::dimension == 1)
            for (std::size_t i = min_; i <= max_; ++i)
                cells += 2;

        if constexpr (Array::dimension == 2)
            for (std::size_t i = min_; i <= max_; ++i)
                cells += (shape[0] - (i * 2) - 2) * 2 + (shape[1] - (i * 2) - 2) * 2 + 4;

        if constexpr (Array::dimension == 3)
            throw std::runtime_error("Not implemented dimension");

        return cells;
    }


    NO_DISCARD auto min() const { return min_; };
    NO_DISCARD auto max() const { return max_; };

private:
    std::size_t min_, max_;
};




template<typename Array, typename Mask>
void operator>>(MaskedView<Array, Mask>&& inner, MaskedView<Array, Mask>&& outer)
{
    using MaskedView_t = MaskedView<Array, Mask>;

    if constexpr (MaskedView_t::dimension == 1)
    {
        assert(inner.xstart() > outer.xstart());
        assert(inner.xend() < outer.xend());
        outer(outer.xstart()) = inner(inner.xstart());
        outer(outer.xend())   = inner(inner.xend());
    }


    if constexpr (MaskedView_t::dimension == 2)
    {
        assert(inner.xstart() > outer.xstart() and inner.xend() < outer.xend()
               and inner.ystart() > outer.ystart() and inner.yend() < outer.yend());

        for (auto ix = inner.xstart(); ix <= inner.xend(); ++ix)
        {
            outer(ix, outer.ystart()) = inner(ix, inner.ystart()); // bottom
            outer(ix, outer.yend())   = inner(ix, inner.yend());   // top
        }

        for (auto iy = inner.ystart(); iy <= inner.yend(); ++iy)
        {
            outer(outer.xstart(), iy) = inner(inner.xstart(), iy); // left
            outer(outer.xend(), iy)   = inner(inner.xend(), iy);   // right
        }

        // bottom left
        for (auto ix = outer.xstart(); ix < inner.xstart(); ++ix)
            outer(ix, outer.ystart()) = inner(inner.xstart(), inner.ystart());

        for (std::size_t iy = outer.ystart(); iy < inner.ystart(); ++iy)
            outer(outer.xstart(), iy) = inner(inner.xstart(), inner.ystart());


        // top left
        for (auto ix = outer.xstart(); ix < inner.xstart(); ++ix)
            outer(ix, outer.yend()) = inner(inner.xstart(), inner.yend());

        for (auto iy = outer.yend(); iy > inner.yend(); --iy)
            outer(outer.xstart(), iy) = inner(inner.xstart(), inner.yend());

        // top right
        for (auto ix = outer.xend(); ix > inner.xend(); --ix)
            outer(ix, outer.yend()) = inner(inner.xend(), inner.yend());

        for (auto iy = outer.yend(); iy > inner.yend(); --iy)
            outer(outer.xend(), iy) = inner(inner.xend(), inner.yend());


        // bottom right
        for (auto ix = outer.xend(); ix > inner.xend(); --ix)
            outer(ix, outer.ystart()) = inner(inner.xend(), inner.ystart());

        for (auto iy = outer.ystart(); iy < inner.ystart(); ++iy)
            outer(outer.xend(), iy) = inner(inner.xend(), inner.ystart());
    }

    if constexpr (MaskedView_t::dimension == 3)
    {
        throw std::runtime_error("3d not implemented");
    }
}

} // namespace PHARE::core

#endif // PHARE_CORE_DATA_NDARRAY_NDARRAY_VECTOR_HPP
