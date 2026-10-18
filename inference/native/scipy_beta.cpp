// Python-free error boundary for the unchanged SciPy/Boost beta quantile body.
#include "bridge.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <boost/math/special_functions/beta.hpp>

namespace {
thread_local std::string beta_warning, beta_error;
using PyGILState_STATE = int;
constexpr int PyExc_RuntimeWarning = 1, PyExc_OverflowError = 2;
int PyGILState_Ensure() { return 0; }
void PyGILState_Release(int) {}
int PyErr_WarnEx(int, const char* message, int) {
    if (!beta_warning.empty()) beta_warning += '\n';
    beta_warning += message;
    return 0;
}
void PyErr_SetString(int, const char* message) { beta_error = message; }
// Original SciPy defaults ignore these four special-function status codes.
enum { SF_ERROR_DOMAIN, SF_ERROR_OVERFLOW, SF_ERROR_UNDERFLOW, SF_ERROR_OTHER };
void sf_error(const char*, int, const char*) {}
}

#include "scipy_beta/original.hpp"

int imagine_beta_ppf(double p, double a, double b, double* output,
                     const char** warning, const char** error) {
    beta_warning.clear();
    beta_error.clear();
    int result = 0;
    try {
        *output = beta_ppf_wrap(p, a, b);
        if (!beta_error.empty()) result = 1; // original pending OverflowError
    } catch (const std::exception& e) {
        beta_error = e.what();
        result = 2;
    }
    *warning = beta_warning.empty() ? nullptr : beta_warning.c_str();
    *error = beta_error.empty() ? nullptr : beta_error.c_str();
    return result;
}
