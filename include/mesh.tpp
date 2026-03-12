#pragma once

namespace Geometry {

    template <typename T>
    PropertyCreationState Mesh::addMeshProperty(MeshComponent property_mesh_component, const std::string& property_name) {
        switch (property_mesh_component) {
        case MeshComponent::Vertex:
            return addMeshProperty<T>(m_per_vertex_property, property_name); // pass vertex map
        case MeshComponent::HalfEdge:
            return addMeshProperty<T>(m_per_half_edge_property, property_name); // pass HalfEdge map
        case MeshComponent::Face:
            return addMeshProperty<T>(m_per_face_property, property_name); // pass face map
        }
        return PropertyCreationState::TypeMismatch; // should not reach here
    }

    template <typename T>
    std::optional<std::reference_wrapper<Property<T>>> Mesh::getMeshProperty(MeshComponent property_mesh_component, const std::string& property_name) {
        switch (property_mesh_component) {
        case MeshComponent::Vertex:
            return getMeshProperty<T>(m_per_vertex_property, property_name); // pass vertex map
        case MeshComponent::HalfEdge:
            return getMeshProperty<T>(m_per_half_edge_property, property_name); // pass halfEdge map
        case MeshComponent::Face:
            return getMeshProperty<T>(m_per_face_property, property_name); // pass face map
        }
        return std::nullopt;
    }

    template <typename T>
    PropertyCreationState Mesh::addMeshProperty(MeshProperty& property_container, const std::string& property_name) {
        size_t target_size = 0;

        if (&property_container == &m_per_vertex_property) {
            target_size = m_vertex_data.position_x.size();
        }
        else if (&property_container == &m_per_face_property) {
            target_size = m_face_data.start_half_edge.size();
        }
        else if (&property_container == &m_per_half_edge_property) {
            target_size = m_half_edge_data.next.size();
        }
        else {
            return PropertyCreationState::TypeMismatch;
        }

        // check existence
        if (property_container.find(property_name) != property_container.end()) {
            return PropertyCreationState::AlreadyExists;
        }

        // create and resize
        std::unique_ptr<BaseProperty> prop(new Property<T>());
        prop->reserve(target_size);
        prop->resize(target_size);

        // store
        property_container[property_name] = std::move(prop);

        return PropertyCreationState::Created;
    }

    template <typename T>
    std::optional<std::reference_wrapper<Property<T>>> Mesh::getMeshProperty(MeshProperty& property_container, const std::string& property_name) {
        auto it = property_container.find(property_name);
        if (it == property_container.end()) return std::nullopt;

        Property<T>* raw_ptr = dynamic_cast<Property<T>*>(it->second.get());
        if (!raw_ptr) return std::nullopt;

        return std::ref(*raw_ptr);
    }

    template <typename T>
    const std::vector<T>& Mesh::getPropertyData(MeshComponent property_mesh_component, const std::string& property_name) {
        auto propeprty_reference = getMeshProperty<T>(property_mesh_component, property_name);
        if (propeprty_reference.has_value()) {
            return propeprty_reference->get().m_data;
        }
        static const std::vector<T> empty_vec;
        return empty_vec;
    }

} 