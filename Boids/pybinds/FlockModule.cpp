/*
 * Created by Simon Idoko on 19.02.2022.
 * */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Flock.h"

namespace py = pybind11;

PYBIND11_MODULE(FlockModule, m) {
    py::class_<Flock>(m, "Flock")
            .def(py::init<>())
            .def(py::init<const Flock &>(), "constructor 2", py::arg("other"))
            .def("__getitem__", &Flock::operator[], py::arg("i"))
            .def("add", &Flock::add, py::arg("boid"))
            .def("clear", &Flock::clear)
            .def("size", &Flock::size)
            .def_property("boids", &Flock::getBoids, &Flock::setBoids);
}