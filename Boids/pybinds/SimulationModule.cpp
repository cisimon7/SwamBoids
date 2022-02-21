/*
 * Created by Ryan Strauss on 11/18/21.
 * */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <Simulation.h>

namespace py = pybind11;

PYBIND11_MODULE(SimulationModule, m) {

    py::class_<Simulation>(m, "Simulation")
            .def(
                    py::init<>(
                            [](int window_width, int window_height, float boid_size, float max_speed, float max_force,
                               float alignment_weight, float cohesion_weight, float separation_weight,
                               float acceleration_scale, float perception, float separation_distance, float noise_scale,
                               bool fullscreen, bool light_scheme, int num_threads, int frame_rate) {
                                return new Simulation(window_width, window_height, boid_size, max_speed, max_force,
                                                      alignment_weight,
                                                      cohesion_weight, separation_weight, acceleration_scale,
                                                      perception,
                                                      separation_distance, noise_scale, fullscreen, light_scheme,
                                                      num_threads, frame_rate);
                            }
                    ),
                    py::arg("window_width") = Simulation::DEFAULT_WINDOW_WIDTH,
                    py::arg("window_height") = Simulation::DEFAULT_WINDOW_HEIGHT,
                    py::arg("boid_size") = Simulation::DEFAULT_BOID_SIZE,
                    py::arg("max_speed") = Simulation::DEFAULT_MAX_SPEED,
                    py::arg("max_force") = Simulation::DEFAULT_MAX_FORCE,
                    py::arg("alignment_weight") = Simulation::DEFAULT_ALIGNMENT_WEIGHT,
                    py::arg("cohesion_weight") = Simulation::DEFAULT_COHESION_WEIGHT,
                    py::arg("separation_weight") = Simulation::DEFAULT_SEPARATION_WEIGHT,
                    py::arg("acceleration_scale") = Simulation::DEFAULT_ACCELERATION_SCALE,
                    py::arg("perception") = Simulation::DEFAULT_PERCEPTION,
                    py::arg("separation_distance") = Simulation::DEFAULT_SEPARATION_DISTANCE,
                    py::arg("noise_scale") = Simulation::DEFAULT_NOISE_SCALE,
                    py::arg("fullscreen") = false,
                    py::arg("light_scheme") = false,
                    py::arg("num_threads") = -1,
                    py::arg("frame_rate") = 60
            )
            .def_property("flock", &Simulation::getFlock, &Simulation::setFlock)
            .def_property("shapes", &Simulation::getShapes, &Simulation::setShapes)
            .def("run", &Simulation::run, py::arg("flock_size"), py::arg("pred_size")=0, py::arg("on_frame"), py::arg("ext_update") = false)
            .def("step_run", &Simulation::step_run, py::arg("flock_size"), py::arg("pred_size")=0, py::arg("on_frame"), py::arg("delay_ms"))
//            .def("add_boid", &Simulation::add_boid, py::arg("x"), py::arg("y"), py::arg("is_predator"), py::arg("with_shape") = true)
            .def("benchmark", &Simulation::benchmark, py::arg("flock_size"), py::arg("num_steps"));

}
