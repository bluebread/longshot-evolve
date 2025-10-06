#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "bool.hpp"

using namespace longshot;
using namespace pybind11::literals;

namespace py = pybind11;

class PyBaseBooleanFunction : public BaseBooleanFunction
{
public:
    using BaseBooleanFunction::BaseBooleanFunction;

    bool eval(BaseBooleanFunction::input_t x) const override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            BaseBooleanFunction,
            eval,
            x
        );
    }
    void as_cnf() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            BaseBooleanFunction,
            as_cnf
        );
    }
    void as_dnf() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            BaseBooleanFunction,
            as_dnf
        );
    }
};


PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<DecisionTree>(m, "_CppDecisionTree")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<int, const DecisionTree &, const DecisionTree &>())
        .def(py::init<const DecisionTree &>())
        .def("delete", &DecisionTree::delete_tree) // remember to delete the tree at the end
        .def("decide", &DecisionTree::decide) 
        .def_property_readonly("lt", &DecisionTree::ltree)
        .def_property_readonly("rt", &DecisionTree::rtree)
        .def_property_readonly("is_constant", &DecisionTree::is_constant)
        .def_property_readonly("var", &DecisionTree::var)
        ;

    py::class_<BaseBooleanFunction, PyBaseBooleanFunction /* <--- trampoline */>(m, "_BaseBooleanFunction")
        .def(py::init<int>())
        .def("eval", &BaseBooleanFunction::eval)
        .def("as_cnf", &BaseBooleanFunction::as_cnf)
        .def("as_dnf", &BaseBooleanFunction::as_dnf)
        .def("avgQ", &BaseBooleanFunction::avgQ, "tree"_a = nullptr)
        .def_property_readonly("num_vars", &BaseBooleanFunction::num_vars)
        ;

    py::class_<MonotonicBooleanFunction, BaseBooleanFunction>(m, "_MonotonicBooleanFunction")
        .def(py::init<int>())
        .def(py::init<const MonotonicBooleanFunction &>())
        .def("eval", &MonotonicBooleanFunction::eval)
        .def("as_cnf", &MonotonicBooleanFunction::as_cnf)
        .def("as_dnf", &MonotonicBooleanFunction::as_dnf)
        .def("avgQ", &MonotonicBooleanFunction::avgQ, "tree"_a = nullptr)
        .def_property_readonly("num_vars", &MonotonicBooleanFunction::num_vars)
        ;

    py::class_<CountingBooleanFunction, BaseBooleanFunction>(m, "_CountingBooleanFunction")
        .def(py::init<int>())
        .def(py::init<const CountingBooleanFunction &>())
        .def("eval", &CountingBooleanFunction::eval)
        .def("as_cnf", &CountingBooleanFunction::as_cnf)
        .def("as_dnf", &CountingBooleanFunction::as_dnf)
        .def("avgQ", &CountingBooleanFunction::avgQ, "tree"_a = nullptr)
        .def_property_readonly("num_vars", &CountingBooleanFunction::num_vars)
        ;
}