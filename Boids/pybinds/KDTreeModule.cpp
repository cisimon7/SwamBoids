/*
 * Created by Simon Idoko on 19.02.2022.
 * */

#include <pybind11/pybind11.h>
#include "KDTree.h"

namespace py = pybind11;

PYBIND11_MODULE(KDTreeModule, m) {

    using NodePtr = std::shared_ptr<Node>;

    py::class_<Node>(m, "Node")
            .def(py::init<Boid *, bool, const NodePtr &, const NodePtr &>(), py::arg("boid"), py::arg("vertical"),
                 py::arg("left"), py::arg("right"))
            .def_readwrite("left", &Node::left)
            .def_readwrite("left", &Node::left)
            .def_readwrite("boid", &Node::boid)
            .def_readwrite("vertical", &Node::vertical);

    py::class_<KDTree>(m, "KDTree")
            .def(py::init<float, float>(), py::arg("width"), py::arg("height"))
            .def("insert", &KDTree::insert, py::arg("boid"))
            .def("search", &KDTree::search, py::arg("query"), py::arg("radius"));
}