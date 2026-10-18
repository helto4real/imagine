// Selected bodies copied verbatim from SciPy 1.17.1. See SCIPY-LICENSE.txt.
template <class RealType>
RealType
boost::math::policies::user_evaluation_error(const char* function, const char* message, const RealType& val) {
    std::string msg("Error in function ");
    std::string haystack {function};
    const std::string needle {"%1%"};
    msg += haystack.replace(haystack.find(needle), needle.length(), typeid(RealType).name()) + ": ";
    // "message" may have %1%, but arguments don't always contain all
    // required information, so don't call boost::format for now
    msg += message;
    PyGILState_STATE save = PyGILState_Ensure();
    PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1);
    PyGILState_Release(save);
    return val;
}

template <class RealType>
RealType
boost::math::policies::user_overflow_error(const char* function, const char* message, const RealType& val) {
    std::string msg("Error in function ");
    std::string haystack {function};
    const std::string needle {"%1%"};
    msg += haystack.replace(haystack.find(needle), needle.length(), typeid(RealType).name()) + ": ";
    // From Boost docs: "overflow and underflow messages do not contain this %1% specifier
    //                   (since the value of value is immaterial in these cases)."
    if (message) {
        msg += message;
    }
    PyGILState_STATE save = PyGILState_Ensure();
    PyErr_SetString(PyExc_OverflowError, msg.c_str());
    PyGILState_Release(save);
    return 0;
}

template<typename Real, typename Policy>
static inline
Real ibeta_inv_wrap(Real a, Real b, Real p, const Policy& policy_)
{
    Real y;

    if (std::isnan(a) || std::isnan(b) || std::isnan(p)) {
        return NAN;
    }
    if ((a <= 0) || (b <= 0) || (p < 0) || (p > 1)) {
        sf_error("betaincinv", SF_ERROR_DOMAIN, NULL);
        return NAN;
    }
    try {
        y = boost::math::ibeta_inv(a, b, p, policy_);
    } catch (const std::domain_error& e) {
        sf_error("betaincinv", SF_ERROR_DOMAIN, NULL);
        y = NAN;
    } catch (const std::overflow_error& e) {
        sf_error("betaincinv", SF_ERROR_OVERFLOW, NULL);
        y = INFINITY;
    } catch (const std::underflow_error& e) {
        sf_error("betaincinv", SF_ERROR_UNDERFLOW, NULL);
        y = 0;
    } catch (...) {
        sf_error("betaincinv", SF_ERROR_OTHER, NULL);
        y = NAN;
    }
    return y;
}

template<typename Real>
Real beta_ppf_wrap(const Real x, const Real a, const Real b)
{
    typedef boost::math::policies::policy<
        boost::math::policies::domain_error<boost::math::policies::ignore_error >,
        boost::math::policies::overflow_error<boost::math::policies::user_error >,
        boost::math::policies::evaluation_error<boost::math::policies::user_error >,
        boost::math::policies::promote_double<false > > BetaPolicyForStats;

    return ibeta_inv_wrap(a, b, x, BetaPolicyForStats());
}
