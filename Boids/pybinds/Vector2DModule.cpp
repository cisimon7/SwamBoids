/*
 * Created by Simon Idoko on 19.02.2022.
 * */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Boid.h"

namespace py = pybind11;

PYBIND11_MODULE(Vector2DModule, m) {
    py::class_<Vector2D>(m, "Vector2D")
            .def(py::init<float, float>(), py::arg("x"), py::arg("y"))
            .def_readwrite("x", &Vector2D::x)
            .def_readwrite("y", &Vector2D::y)
            .def("__repr__",
                 [](const Vector2D &vec) {
                     return "Vector2D(x=" + std::to_string(vec.x) + ", y=" + std::to_string(vec.y) + ")";
                 }
            ).def(py::pickle(
                    [](const Vector2D &vec) {
                        return py::make_tuple(vec.x, vec.y);
                    },
                    [](py::tuple t) {
                        if (t.size() != 2)
                            throw std::runtime_error("Invalid Vector2D Pickle State!");

                        return Vector2D(t[0].cast<float>(), t[1].cast<float>());
                    }
            ));
}

