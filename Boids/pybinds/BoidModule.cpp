/*
 * Created by Simon Idoko on 19.02.2022.
 * */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Boid.h"

namespace py = pybind11;

PYBIND11_MODULE(BoidModule, m) {
    py::class_<Boid>(m, "Boid")
            .def(py::init<>(
                         [](int boid_id, float x, float y, float max_width, float max_height, float max_speed,
                            float max_force,
                            float acceleration_scale, float cohesion_weight, float alignment_weight,
                            float separation_weight, float perception, float separation_distance, float noise_scale,
                            bool is_predator = false) {
                             return new Boid(boid_id, x, y, max_width, max_height, max_speed, max_force, acceleration_scale,
                                             cohesion_weight, alignment_weight, separation_weight,
                                             perception, separation_distance, noise_scale, is_predator);
                         }
                 ),
                 py::arg("boid_id"),
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
            .def("angle", &Boid::angle)
//            .def_property_readonly("boid_id", &Boid::getBoidId)
            .def_readwrite("boid_id", &Boid::boid_id)
            .def_readwrite("position", &Boid::position)
            .def_readwrite("velocity", &Boid::velocity)
            .def_readwrite("acceleration", &Boid::acceleration)
            .def_readwrite("max_width", &Boid::max_width)
            .def_readwrite("max_height", &Boid::max_height)
            .def_readwrite("max_speed", &Boid::max_speed)
            .def_readwrite("max_force", &Boid::max_force)
            .def_readwrite("acceleration_scale", &Boid::acceleration_scale)
            .def_readwrite("cohesion_weight", &Boid::cohesion_weight)
            .def_readwrite("alignment_weight", &Boid::alignment_weight)
            .def_readwrite("separation_weight", &Boid::separation_weight)
            .def_readwrite("perception", &Boid::perception)
            .def_readwrite("separation_distance", &Boid::separation_distance)
            .def_readwrite("noise_scale", &Boid::noise_scale)
            .def_readwrite("is_predator", &Boid::is_predator)
            .def("__repr__",
                 [](const Boid &boid) {
                     return "Boid(x=" + std::to_string(boid.position.x) + ", y=" + std::to_string(boid.position.y) +
                            ")";
                 }
            )
            .def(py::pickle(
                    [](const Boid &p) { // __getstate__
                        return py::make_tuple(
                                p.boid_id, // 0
                                p.position,// 1
                                p.max_width,// 2
                                p.max_height,// 3
                                p.max_speed,// 4
                                p.max_force,// 5
                                p.acceleration_scale,// 6
                                p.cohesion_weight,// 7
                                p.alignment_weight,// 8
                                p.separation_weight,// 9
                                p.perception,// 10
                                p.separation_distance,// 11
                                p.noise_scale,// 12
                                p.is_predator);// 13
                    },
                    [](py::tuple t) { // __setstate__
                        if (t.size() != 14)
                            throw std::runtime_error("Invalid Boid Pickle state!");

                        Boid boid(
                                t[0].cast<int>(),
                                t[1].cast<Vector2D>().x,
                                t[1].cast<Vector2D>().y,
                                t[2].cast<float>(),
                                t[3].cast<float>(),
                                t[4].cast<float>(),
                                t[5].cast<float>(),
                                t[6].cast<float>(),
                                t[7].cast<float>(),
                                t[8].cast<float>(),
                                t[9].cast<float>(),
                                t[10].cast<float>(),
                                t[11].cast<float>(),
                                t[12].cast<float>(),
                                t[13].cast<bool>()
                        );

                        return boid;
                    }
            ));
}