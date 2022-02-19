/*Created by Simon Idoko on 18.02.2022.*/

#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

struct Triangle {
private:
    float width, height;

public:
    Triangle(const float width, const float height) : width(width), height(height) {}

    double area() const {
        return 0.5 * height * width;
    }
};

PYBIND11_MODULE(Geometry, m) {
    std::cout << "The module Geometry has been loaded !" << std::endl;

    py::class_<Triangle> t(m, "Triangle");
    t.def(py::init<const float &, const float &>());
    t.def("area", &Triangle::area);

}

