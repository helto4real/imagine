/* Python-free, thread-local callback boundary to unchanged SciPy QUADPACK. */
#include <stddef.h>
void dqagse(double (*)(double*), double, double, double, double, int,
            double*, double*, int*, int*, double*, double*, double*, double*, int*, int*);
typedef double (*imagine_integrand)(double, void*);
static _Thread_local imagine_integrand current_callback;
static _Thread_local void* current_context;
static double callback(double* x) { return current_callback(*x, current_context); }
void at_imagine_quadpack(imagine_integrand integrand, void* context,
                        double a, double b, double* result, double* error,
                        int* evaluations, int* status) {
    imagine_integrand previous_callback = current_callback;
    void* previous_context = current_context;
    current_callback = integrand;
    current_context = context;
    double alist[50], blist[50], rlist[50], elist[50];
    int iord[50], last;
    dqagse(callback, a, b, 1.49e-8, 1e-4, 50, result, error, evaluations,
           status, alist, blist, rlist, elist, iord, &last);
    current_callback = previous_callback;
    current_context = previous_context;
}
