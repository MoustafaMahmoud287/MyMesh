#pragma once
#include <cmath>
#include <iostream>
#include <cassert>
#include <limits> 

struct Point {
    float x, y, z;

    // constructors 
    Point() : x(0), y(0), z(0) {}
    Point(float x, float y, float z) : x(x), y(y), z(z) {}

    // equalitiy 
    bool operator==(const Point& other) const {
        const float epsilon = 1e-6f;
        return std::abs(x - other.x) < epsilon &&
            std::abs(y - other.y) < epsilon &&
            std::abs(z - other.z) < epsilon;
    }

    bool operator!=(const Point& other) const {
        return !(*this == other);
    }

    // basic arithmetic
    Point operator+(const Point& other) const { return Point(x + other.x, y + other.y, z + other.z); }
    Point operator-(const Point& other) const { return Point(x - other.x, y - other.y, z - other.z); }
    Point operator*(float s) const { return Point(x * s, y * s, z * s); }

    Point operator/(float s) const {
        assert(std::abs(s) > 1e-8f && "Division by zero!");
        return Point(x / s, y / s, z / s);
    }

    Point operator-() const {
        return Point(-x, -y, -z);
    }

    // compound assignment
    Point& operator+=(const Point& other) { x += other.x; y += other.y; z += other.z; return *this; }
    Point& operator-=(const Point& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    Point& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    Point& operator/=(float s) {
        assert(std::abs(s) > 1e-8f && "Division by zero!");
        x /= s; y /= s; z /= s;
        return *this;
    }

    // vector operations
    float dot(const Point& other) const { return x * other.x + y * other.y + z * other.z; }

    Point cross(const Point& other) const {
        return Point(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    float lengthSquared() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(lengthSquared()); }

    Point normalize() const {
        float len = length();
        if (len > 1e-8f) return *this / len; 
        return Point(0, 0, 0);
    }

    // array access
    float& operator[](int i) {
        assert(i >= 0 && i < 3 && "Index out of bounds!");
        return (&x)[i];
    }

    const float& operator[](int i) const {
        assert(i >= 0 && i < 3 && "Index out of bounds!");
        return (&x)[i];
    }

    // stream Printing 
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
    }
};

inline Point operator*(float s, const Point& v) { return v * s; }

inline float dot(const Point& p1, const Point& p2) { return p1.dot(p2); }

inline Point cross(Point& p1, const Point& p2) { return p1.cross(p2); }