/*
 * Created by Simon Idoko on 19.02.2022.
 * */

#include <pybind11/pybind11.h>
#include "Boid.h"

namespace py = pybind11;

PYBIND11_MODULE(BoidModule, m) {
    py::class_<Boid>(m, "Boid")
            .def(py::init<>(
                         [](float x, float y, float max_width, float max_height, float max_speed, float max_force,
                            float acceleration_scale, float cohesion_weight, float alignment_weight,
                            float separation_weight, float perception, float separation_distance, float noise_scale,
                            bool is_predator = false) {
                             return new Boid(x, y, max_width, max_height, max_speed, max_force, acceleration_scale,
                                             cohesion_weight, alignment_weight, separation_weight,
                                             perception, separation_distance, noise_scale, is_predator);
                         }
                 ),
                 py::arg("x"),
                 py::arg("y"),
                 py::arg("max_width"),
                 py::arg("max_height"),
                 py::arg("max_speed"),
                 py::arg("max_force"),
                 py::arg("acceleration_scale"),
                 py::arg("cohesion_weight"),
                 py::arg("alignment_weight"),
                 py::arg("separation_weight"),
                 py::arg("perception"),
                 py::arg("separation_distance"),
                 py::arg("noise_scale"),
                 py::arg("is_predator") = false
            )
            .def(py::init<const Boid &>(), "constructor 2")
            .def("assign", &Boid::operator=, py::arg("boids"))
            .def("update", &Boid::update, py::arg("boids"))
            .def("angle", &Boid::angle);
}