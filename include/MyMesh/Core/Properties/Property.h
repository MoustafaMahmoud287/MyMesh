#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm> 
#include <memory>
#include <cassert> 

namespace Geometry {

    class Mesh; // forward declaration

    // --- the abstract interface ---
    // Allows Mesh to manage properties without knowing their type
    class BaseProperty {
        friend class Mesh; // Mesh manages the lifecycle
    public:
        virtual ~BaseProperty() = default;

    protected:
        BaseProperty() = default;

        // Pure virtual functions for synchronization
        virtual std::unique_ptr<BaseProperty> clone() const = 0;
        virtual void reserve(size_t n) = 0;
        virtual void resize(size_t n) = 0;
        virtual void clear() = 0;
        virtual void push_back() = 0;
    };

    // --- The Concrete Implementation ---
    template <typename T>
    class Property : public BaseProperty {
        friend class Mesh; // mesh needs access to m_data

    protected:
        // protected constructor: 
        // only mesh can create properties to ensure they are synchronized.
        Property() = default;

        // copy logic for Mesh copying
        virtual std::unique_ptr<BaseProperty> clone() const override {
            return std::unique_ptr<BaseProperty>(new Property<T>(*this));
        }

        // Synchronization overrides
        void reserve(size_t n) override { m_data.reserve(n); }
        void resize(size_t n) override { m_data.resize(n); }
        void clear() override { m_data.clear(); }

        // When Mesh adds a vertex, we add a default T() here
        void push_back() override { m_data.push_back(T()); }

    public:
        // public accessors

        T& operator[](size_t idx) {
            assert(idx < m_data.size() && "Property index out of bounds!");
            return m_data[idx];
        }

        const T& operator[](size_t idx) const {
            assert(idx < m_data.size() && "Property index out of bounds!");
            return m_data[idx];
        }

        size_t size() const { return m_data.size(); }

    private:
        // the actual data storage
        std::vector<T> m_data;
    };

} 