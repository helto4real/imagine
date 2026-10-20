/*
 * Copyright (C) 2024 SciPy developers
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * b. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * c. Names of the SciPy Developers may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
static void dqelg(int*,double*,double*,double*,double*,int*);
static void dqk21(double(*)(double*),const double,const double,double*,double*,double*,double*);
static void dqpsrt(const int,const int,int*,double*,const double*,int*,int*);

// Constants
static const double uflow = 2.2250738585072014e-308;  /* np.finfo(np.float64).tiny */
static const double oflow = 1.7976931348623157e+308;  /* np.finfo(np.float64).max  */
static const double epmach = 2.220446049250313e-016;  /* np.finfo(np.float64).eps  */




void
dqagse(double(*fcn)(double* x), const double a, const double b,
       const double epsabs, const double epsrel, const int limit, double* result,
       double* abserr, int* neval, int* ier, double* alist, double* blist,
       double* rlist, double* elist, int* iord, int* last)
{
    // ***begin prologue  dqagse
    // ***date written   800101   (yymmdd)
    // ***revision date  830518   (yymmdd)
    // ***category no.  h2a1a1
    // ***keywords  automatic integrator, general-purpose,
    //              (end point) singularities, extrapolation,
    //              globally adaptive
    // ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
    //            de doncker,elise,appl. math. & progr. div. - k.u.leuven
    // ***purpose  the routine calculates an approximation result to a given
    //             definite integral i = integral of f over (a,b),
    //             hopefully satisfying following claim for accuracy
    //             abs(i-result).le.max(epsabs,epsrel*abs(i)).
    // ***description
    //
    //         computation of a definite integral
    //         standard fortran subroutine
    //         double precision version
    //
    //         parameters
    //          on entry
    //             f      - double precision
    //                      function subprogram defining the integrand
    //                      function f(x). the actual name for f needs to be
    //                      declared e x t e r n a l in the driver program.
    //
    //             a      - double precision
    //                      lower limit of integration
    //
    //             b      - double precision
    //                      upper limit of integration
    //
    //             epsabs - double precision
    //                      absolute accuracy requested
    //             epsrel - double precision
    //                      relative accuracy requested
    //                      if  epsabs.le.0
    //                      and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
    //                      the routine will end with ier = 6.
    //
    //             limit  - integer
    //                      gives an upperbound on the number of subintervals
    //                      in the partition of (a,b)
    //
    //          on return
    //             result - double precision
    //                      approximation to the integral
    //
    //             abserr - double precision
    //                      estimate of the modulus of the absolute error,
    //                      which should equal or exceed abs(i-result)
    //
    //             neval  - integer
    //                      number of integrand evaluations
    //
    //             ier    - integer
    //                      ier = 0 normal and reliable termination of the
    //                              routine. it is assumed that the requested
    //                              accuracy has been achieved.
    //                      ier.gt.0 abnormal termination of the routine
    //                              the estimates for integral and error are
    //                              less reliable. it is assumed that the
    //                              requested accuracy has not been achieved.
    //             error messages
    //                          = 1 maximum number of subdivisions allowed
    //                              has been achieved. one can allow more sub-
    //                              divisions by increasing the value of limit
    //                              (and taking the according dimension
    //                              adjustments into account). however, if
    //                              this yields no improvement it is advised
    //                              to analyze the integrand in order to
    //                              determine the integration difficulties. if
    //                              the position of a local difficulty can be
    //                              determined (e.g. singularity,
    //                              discontinuity within the interval) one
    //                              will probably gain from splitting up the
    //                              interval at this point and calling the
    //                              integrator on the subranges. if possible,
    //                              an appropriate special-purpose integrator
    //                              should be used, which is designed for
    //                              handling the type of difficulty involved.
    //                          = 2 the occurrence of roundoff error is detec-
    //                              ted, which prevents the requested
    //                              tolerance from being achieved.
    //                              the error may be under-estimated.
    //                          = 3 extremely bad integrand behaviour
    //                              occurs at some points of the integration
    //                              interval.
    //                          = 4 the algorithm does not converge.
    //                              roundoff error is detected in the
    //                              extrapolation table.
    //                              it is presumed that the requested
    //                              tolerance cannot be achieved, and that the
    //                              returned result is the best which can be
    //                              obtained.
    //                          = 5 the integral is probably divergent, or
    //                              slowly convergent. it must be noted that
    //                              divergence can occur with any other value
    //                              of ier.
    //                          = 6 the input is invalid, because
    //                              epsabs.le.0 and
    //                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28).
    //                              result, abserr, neval, last, rlist(1),
    //                              iord(1) and elist(1) are set to zero.
    //                              alist(1) and blist(1) are set to a and b
    //                              respectively.
    //
    //             alist  - double precision
    //                      vector of dimension at least limit, the first
    //                       last  elements of which are the left end points
    //                      of the subintervals in the partition of the
    //                      given integration range (a,b)
    //
    //             blist  - double precision
    //                      vector of dimension at least limit, the first
    //                       last  elements of which are the right end points
    //                      of the subintervals in the partition of the given
    //                      integration range (a,b)
    //
    //             rlist  - double precision
    //                      vector of dimension at least limit, the first
    //                       last  elements of which are the integral
    //                      approximations on the subintervals
    //
    //             elist  - double precision
    //                      vector of dimension at least limit, the first
    //                       last  elements of which are the moduli of the
    //                      absolute error estimates on the subintervals
    //
    //             iord   - integer
    //                      vector of dimension at least limit, the first k
    //                      elements of which are pointers to the
    //                      error estimates over the subintervals,
    //                      such that elist(iord(1)), ..., elist(iord(k))
    //                      form a decreasing sequence, with k = last
    //                      if last.le.(limit/2+2), and k = limit+1-last
    //                      otherwise
    //
    //             last   - integer
    //                      number of subintervals actually produced in the
    //                      subdivision process
    //
    // ***references  (none)
    // ***routines called  d1mach,dqelg,dqk21,dqpsrt
    // ***end prologue  dqagse
    //
    //
    //             the dimension of rlist2 is determined by the value of
    //             limexp in subroutine dqelg (rlist2 should be of dimension
    //             (limexp+2) at least).
    //
    //             list of major variables
    //             -----------------------
    //
    //            alist     - list of left end points of all subintervals
    //                        considered up to now
    //            blist     - list of right end points of all subintervals
    //                        considered up to now
    //            rlist(i)  - approximation to the integral over
    //                        (alist(i),blist(i))
    //            rlist2    - array of dimension at least limexp+2 containing
    //                        the part of the epsilon table which is still
    //                        needed for further computations
    //            elist(i)  - error estimate applying to rlist(i)
    //            maxerr    - pointer to the interval with largest error
    //                        estimate
    //            errmax    - elist(maxerr)
    //            erlast    - error on the interval currently subdivided
    //                        (before that subdivision has taken place)
    //            area      - sum of the integrals over the subintervals
    //            errsum    - sum of the errors over the subintervals
    //            errbnd    - requested accuracy max(epsabs,epsrel*
    //                        abs(result))
    //            *****1    - variable for the left interval
    //            *****2    - variable for the right interval
    //            last      - index for subdivision
    //            nres      - number of calls to the extrapolation routine
    //            numrl2    - number of elements currently in rlist2. if an
    //                        appropriate approximation to the compounded
    //                        integral has been obtained it is put in
    //                        rlist2(numrl2) after numrl2 has been increased
    //                        by one.
    //            small     - length of the smallest interval considered up
    //                        to now, multiplied by 1.5
    //            erlarg    - sum of the errors over the intervals larger
    //                        than the smallest interval considered up to now
    //            extrap    - logical variable denoting that the routine is
    //                        attempting to perform extrapolation i.e. before
    //                        subdividing the smallest interval we try to
    //                        decrease the value of erlarg.
    //            noext     - logical variable denoting that extrapolation
    //                        is no longer allowed (true value)
    //
    //             machine dependent constants
    //             ---------------------------
    //
    //            epmach is the largest relative spacing.
    //            uflow is the smallest positive magnitude.
    //            oflow is the largest positive magnitude.
    //
    int extrap, ierror, iroff1, iroff2, iroff3, k, ksgn, ktmin, L, maxerr;
    int noext, nres, nrmax, numrl2;
    double a1, a2, area, area1, area12, area2, b1, b2, defab1, defab2, defabs;
    double dres, erlarg, erlast, errbnd, errmax, error1, error12, error2, errsum;
    double ertest, small, jupbnd, correc, resabs, reseps, abseps;
    double res3la[3], rlist2[52];
    small = 0.0;

    *ier = 0;
    *neval = 0;
    *last = 0;
    alist[0] = a;
    blist[0] = b;
    rlist[0] = 0.0;
    elist[0] = 0.0;
    iord[0] = 0;
    *result = 0.0;
    *abserr = 0.0;
    ierror = 0;
    erlarg = 0.0;
    ertest = 0.0;
    correc = 0.0;

    if ((epsabs <= 0.0) && (epsrel < fmax(50.0*epmach, 0.5e-28)))
    {
        *ier = 6;
        return;
    }

    // First approximation to the integral.
    dqk21(fcn, a, b, result, abserr, &defabs, &resabs);

    // Test on accuracy
    dres = fabs(*result);
    errbnd = fmax(epsabs, epsrel*dres);
    *last = 1;
    rlist[0] = *result;
    elist[0] = *abserr;
    iord[0] = 0;

    if ((*abserr <= 100.0*epmach*defabs) && (*abserr > errbnd)) { *ier = 2; }
    if (limit == 1) { *ier = 1; }
    if ((*ier != 0) || ((*abserr <= errbnd) && (*abserr != resabs)) || (*abserr == 0.0)) { goto LINE140; }

    // Initialization.
    rlist2[0] = *result;
    errmax = *abserr;
    maxerr = 0;
    area = *result;
    errsum = *abserr;
    *abserr = oflow;
    nrmax = 0;
    nres = 0;
    numrl2 = 1;
    ktmin = 0;
    extrap = 0;
    noext = 0;
    iroff1 = 0;
    iroff2 = 0;
    iroff3 = 0;
    ksgn = (dres >= (1.0 - 50.0*epmach)*defabs ? 1 : -1);

    // Main for-loop
    for (L = 1; L < limit; L++)
    {
        *last = L + 1;

        // Bisect the subinterval with nrmax-th largest error estimate.
        a1 = alist[maxerr];
        b1 = 0.5*(alist[maxerr] + blist[maxerr]);
        a2 = b1;
        b2 = blist[maxerr];
        erlast = errmax;
        dqk21(fcn, a1, b1, &area1, &error1, &resabs, &defab1);
        dqk21(fcn, a2, b2, &area2, &error2, &resabs, &defab2);

        // Improve previous approximations to integral and error and test for accuracy.
        area12 = area1 + area2;
        error12 = error1 + error2;
        errsum = errsum + error12 - errmax;
        area = area + area12 - rlist[maxerr];

        if ((defab1 != error1) && (defab2 != error2))
        {
            if (!((fabs(rlist[maxerr] - area12) > 1.0e-5*fabs(area12)) || (error12 < 0.99*errmax)))
            {
                if (extrap) { iroff2++; } else { iroff1++; }
            }
            // 10
            if ((L > 9) && (error12 > errmax)) { iroff3++; }
        }
        // 15

        rlist[maxerr] = area1;
        rlist[L] = area2;
        errbnd = fmax(epsabs, epsrel*fabs(area));

        // Test for roundoff error and eventually set error flag.
        if (((iroff1 + iroff2) >= 10) || (iroff3 >= 20)) { *ier = 2; }
        if (iroff2 >= 5) { ierror = 3; }

        // Set error flag in the case that the number of subintervals equals limit.
        if (*last == limit) { *ier = 1; }

        // Set error flag in the case of bad integrand behavior at some points
        // in the integration range.
        if (fmax(fabs(a1), fabs(b2)) <= (1.0 + 100.0*epmach)*(fabs(a2) + 1000.0*uflow))
        {
            *ier = 4;
        }

        // Append the newly-created intervals to the list.
        if (!(error2 > error1))
        {
            alist[L] = a2;
            blist[maxerr] = b1;
            blist[L] = b2;
            elist[maxerr] = error1;
            elist[L] = error2;
        } else {
            alist[maxerr] = a2;
            alist[L] = a1;
            blist[L] = b1;
            rlist[maxerr] = area2;
            rlist[L] = area1;
            elist[maxerr] = error2;
            elist[L] = error1;
        }

        // call subroutine dqpsrt to maintain the descending ordering in the
        // list of error estimates and select the subinterval with nrmax-th
        // largest error estimate (to be bisected next).
        dqpsrt(limit, *last, &maxerr, &errmax, elist, iord, &nrmax);

        if (errsum <= errbnd) { goto LINE115; }
        if (*ier != 0) { break; }
        if (L == 1) { goto LINE80; }
        if (noext) { continue; }
        erlarg = erlarg - erlast;
        if (fabs(b1 - a1) > small) { erlarg = erlarg + error12; }
        if (!(extrap))
        {
            // Test whether the interval to be bisected next is the smallest interval.
            if (fabs(blist[maxerr] - alist[maxerr]) > small) { continue; }
            extrap = 1;
            nrmax = 1;
        }
        // 40
        if ((ierror == 3) || (erlarg <= ertest)) { goto LINE60; }

        // The smallest interval has the largest error. Before bisecting
        // decrease the sum of the errors over the larger intervals (erlarg)
        // and perform extrapolation.
        jupbnd = (*last > 2 + (limit/2) ? limit + 3 - *last : *last);
        for (k = nrmax; k < jupbnd; k++) {
            maxerr = iord[nrmax];
            errmax = elist[maxerr];
            if (fabs(blist[maxerr] - alist[maxerr]) > small) { goto LINE90; } // break->continue
            nrmax++;
        }
        // 50
LINE60:
        // Perform extrapolation.
        numrl2++;
        rlist2[numrl2] = area;
        dqelg(&numrl2, rlist2, &reseps, &abseps, res3la, &nres);
        ktmin += 1;
        if ((ktmin > 5) && (*abserr < 1.0e-3 * errsum)) { *ier = 5; }
        if (abseps >= *abserr) { goto LINE70; }
        ktmin = 0;
        *abserr = abseps;
        *result = reseps;
        correc = erlarg;
        ertest = fmax(epsabs, epsrel*fabs(reseps));
        if (*abserr <= ertest) { break; }
LINE70:
        // Prepare bisection of the smallest interval.
        if (numrl2 == 0) { noext = 1; }
        if (*ier == 5) { break; }
        maxerr = iord[0];
        errmax = elist[maxerr];
        nrmax = 0;
        extrap = 0;
        small = small * 0.5;
        erlarg = errsum;
        continue;
LINE80:
        small = fabs(b - a)*0.375;
        erlarg = errsum;
        ertest = errbnd;
        rlist2[1] = area;
LINE90:
        ; // no-op
    }

    // Set final result and error estimate
    if (*abserr == oflow) { goto LINE115; }
    if ((*ier + ierror) == 0) { goto LINE110; }
    if (ierror == 3) { *abserr = *abserr + correc; }
    if (*ier == 0) { *ier = 3; }
    if ((*result != 0.0) && (area != 0.0)) { goto LINE105; }
    if (*abserr > errsum) { goto LINE115; }
    if (area == 0.0) { goto LINE130; }
    goto LINE110;
LINE105:
    if (*abserr/fabs(*result) > errsum/fabs(area)) { goto LINE115; }
LINE110:
    // Test on divergence.
    if ((ksgn == -1) && (fmax(fabs(*result), fabs(area)) <= defabs*0.01)) { goto LINE130; }
    if ((0.01 > (*result/area)) || ((*result/area) > 100.0) || (errsum > fabs(area))) { *ier = 6; }
    goto LINE130;

LINE115:
    // Compute global integral sum.
    *result = 0.0;
    for (k = 0; k <= L; k++)
    {
        *result = *result + rlist[k];
    }
    // 120
    *abserr = errsum;
LINE130:
    if (*ier > 2) { *ier -= 1; }
LINE140:
    *neval = 42*(*last) - 21;

    return;
}

void
dqelg(int* n, double* epstab, double* result, double* abserr, double* res3la, int* nres)
{
    // ***begin prologue  dqelg
    // ***refer to  dqagie,dqagoe,dqagpe,dqagse
    // ***routines called  d1mach
    // ***revision date  830518   (yymmdd)
    // ***keywords  epsilon algorithm, convergence acceleration,
    //              extrapolation
    // ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
    //            de doncker,elise,appl. math & progr. div. - k.u.leuven
    // ***purpose  the routine determines the limit of a given sequence of
    //             approximations, by means of the epsilon algorithm of
    //             p.wynn. an estimate of the absolute error is also given.
    //             the condensed epsilon table is computed. only those
    //             elements needed for the computation of the next diagonal
    //             are preserved.
    // ***description
    //
    //            epsilon algorithm
    //            standard fortran subroutine
    //            double precision version
    //
    //            parameters
    //               n      - integer
    //                        epstab(n) contains the new element in the
    //                        first column of the epsilon table.
    //
    //               epstab - double precision
    //                        vector of dimension 52 containing the elements
    //                        of the two lower diagonals of the triangular
    //                        epsilon table. the elements are numbered
    //                        starting at the right-hand corner of the
    //                        triangle.
    //
    //               result - double precision
    //                        resulting approximation to the integral
    //
    //               abserr - double precision
    //                        estimate of the absolute error computed from
    //                        result and the 3 previous results
    //
    //               res3la - double precision
    //                        vector of dimension 3 containing the last 3
    //                        results
    //
    //               nres   - integer
    //                        number of calls to the routine
    //                        (should be zero at first call)
    //
    // ***end prologue  dqelg
    //
    //            list of major variables
    //            -----------------------
    //
    //            e0     - the 4 elements on which the computation of a new
    //            e1       element in the epsilon table is based
    //            e2
    //            e3                 e0
    //                         e3    e1    new
    //                               e2
    //            newelm - number of elements to be computed in the new
    //                     diagonal
    //            error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
    //            result - the element in the new diagonal with least value
    //                     of error
    //
    //            machine dependent constants
    //            ---------------------------
    //
    //            epmach is the largest relative spacing.
    //            oflow is the largest positive magnitude.
    //            limexp is the maximum number of elements the epsilon
    //            table can contain. if this number is reached, the upper
    //            diagonal of the epsilon table is deleted.
    //
    int i, indx, k1, k2, k3, limexp, n2_rem, newelm, starting_n;
    double delta1, delta2, delta3, epsinf, error, err1, err2, err3;
    double e0, e1, e1abs, e2, e3, res, ss, tol1, tol2, tol3;

    // n is an index of epstab hence must be 0-indexed but the algorithm here
    // is tricky in terms of jump conditions. Hence np1 = n + 1 will be used for
    // those parts that require 1-indexed logic.

    starting_n = *n;
    *nres += 1;
    *abserr = oflow;
    *result = epstab[*n];
    if (*n < 2)
    {
        *abserr = fmax(*abserr, 5.0*epmach*fabs(*result));
        return;
    }

    limexp = 49; // indices 50 and 51 for the lower diagonals
    epstab[*n + 2] = epstab[*n];
    newelm = *n / 2;
    epstab[*n] = oflow;
    k1 = *n;
    for (i = 0; i < newelm; i++)
    {
        k2 = k1 - 1;
        k3 = k1 - 2;
        res = epstab[k1 + 2];
        e0 = epstab[k3];
        e1 = epstab[k2];
        e2 = res;
        e1abs = fabs(e1);
        delta2 = e2 - e1;
        err2 = fabs(delta2);
        tol2 = fmax(fabs(e2), e1abs)*epmach;
        delta3 = e1 - e0;
        err3 = fabs(delta3);
        tol3 = fmax(e1abs, fabs(e0))*epmach;
        if (!((err2 > tol2) || (err3 > tol3)))
        {
            // If e0, e1 and e2 are equal to within machine accuracy,
            // convergence is assumed.
            // result = e2
            // abserr = abs(e1-e0)+abs(e2-e1)
            *result = res;
            *abserr = err2 + err3;
            *abserr = fmax(*abserr, 5.0*epmach*fabs(*result));
            return;
        }
        // 10

        e3 = epstab[k1];
        epstab[k1] = e1;
        delta1 = e1 - e3;
        err1 = fabs(delta1);
        tol1 = fmax(e1abs, fabs(e3))*epmach;

        // if two elements are very close to each other, omit a part of the
        // table by adjusting the value of n.
        if ((err1 <= tol1) || (err2 <= tol2) || (err3 <= tol3))
        {
            *n = i + i;
            break;  // Goto 50
        }
        ss = 1.0/delta1 + 1.0/delta2 - 1.0/delta3;
        epsinf = fabs(ss*e1);

        // Test to detect irregular behaviour in the table, and  eventually omit
        // a part of the table adjusting the value of n.
        if (!(epsinf > 1e-4))
        {
            *n = i + i;
            break;  // Goto 50
        }
        // 30

        // Compute a new element and eventually adjust the value of result.
        res = e1 + 1.0 / ss;
        epstab[k1] = res;
        k1 -= 2;
        error = err2 + fabs(res - e2) + err3;
        if (!(error > *abserr))
        {
            *abserr = error;
            *result = res;
        }
    }
    //40

    // Shift the table
    // 50
    if (*n == limexp) { *n = 2*(limexp / 2); }

    n2_rem = (starting_n % 2);
    for (i = 0; i <= newelm; i++)
    {
        epstab[2*i + n2_rem] = epstab[2*i + 2 + n2_rem];
    }
    // 60

    if (*n != starting_n)
    {
        indx = starting_n - *n;
        for (i = 0; i <= *n; i++)
        {
            epstab[i] = epstab[indx];
            indx++;
        }
        // 70
    }
    // 80
    if (*nres < 4)
    {
        res3la[*nres - 1] = *result;
        *abserr = oflow;
    } else {
        // 90
        *abserr = fabs(*result-res3la[2]) + fabs(*result-res3la[1]) + fabs(*result-res3la[0]);
        res3la[0] = res3la[1];
        res3la[1] = res3la[2];
        res3la[2] = *result;
    }
    // 100

    *abserr = fmax(*abserr, 5.0*epmach*fabs(*result));

    return;
}

void
dqk21(double(*fcn)(double* x), const double a, const double b,
      double* result, double* abserr, double* resabs, double* resasc)
{
    // ***begin prologue  dqk21
    // ***date written   800101   (yymmdd)
    // ***revision date  830518   (yymmdd)
    // ***category no.  h2a1a2
    // ***keywords  21-point gauss-kronrod rules
    // ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
    //            de doncker,elise,appl. math. & progr. div. - k.u.leuven
    // ***purpose  to compute i = integral of f over (a,b), with error
    //                            estimate
    //                        j = integral of abs(f) over (a,b)
    // ***description
    //
    //            integration rules
    //            standard fortran subroutine
    //            double precision version
    //
    //            parameters
    //             on entry
    //               f      - double precision
    //                        function subprogram defining the integrand
    //                        function f(x). the actual name for f needs to be
    //                        declared e x t e r n a l in the driver program.
    //
    //               a      - double precision
    //                        lower limit of integration
    //
    //               b      - double precision
    //                        upper limit of integration
    //
    //             on return
    //               result - double precision
    //                        approximation to the integral i
    //                        result is computed by applying the 21-point
    //                        kronrod rule (resk) obtained by optimal addition
    //                        of abscissae to the 10-point gauss rule (resg).
    //
    //               abserr - double precision
    //                        estimate of the modulus of the absolute error,
    //                        which should not exceed abs(i-result)
    //
    //               resabs - double precision
    //                        approximation to the integral j
    //
    //               resasc - double precision
    //                        approximation to the integral of abs(f-i/(b-a))
    //                        over (a,b)
    //
    // ***references  (none)
    // ***routines called  d1mach
    // ***end prologue  dqk21
    //
    //            the abscissae and weights are given for the interval (-1,1).
    //            because of symmetry only the positive abscissae and their
    //            corresponding weights are given.
    //
    //            xgk    - abscissae of the 21-point kronrod rule
    //                     xgk(2), xgk(4), ...  abscissae of the 10-point
    //                     gauss rule
    //                     xgk(1), xgk(3), ...  abscissae which are optimally
    //                     added to the 10-point gauss rule
    //
    //            wgk    - weights of the 21-point kronrod rule
    //
    //            wg     - weights of the 10-point gauss rule
    //
    //
    //  gauss quadrature weights and kronron quadrature abscissae and weights
    //  as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
    //  bell labs, nov. 1981.
    //
    int j, jtw, jtwm1;
    double absc, centr, dhlgth, fc, fsum, fval1, fval2, hlgth, resg, resk, reskh;
    double fv1[10], fv2[10];
    double mut_D;

    static const double wg[5] = {
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338
    };
    static const double xgk[11] = {
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0.000000000000000000000000000000000
    };
    static const double wgk[11] = {
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821
    };

    centr = 0.5 * (a + b);
    hlgth = 0.5 * (b - a);
    dhlgth = fabs(hlgth);

    // Compute the 21-point kronrod approximation to the integral, and
    // estimate the absolute error.

    fc=(*fcn)(&centr);
    resg = 0.0;
    resk = wgk[10]*fc;
    *resabs = fabs(resk);

    for (j = 0; j < 5; j++) {
        jtw = 2*j + 1;
        absc = hlgth * xgk[jtw];

        mut_D = centr - absc;
        fval1 = (*fcn)(&mut_D);
        mut_D = centr + absc;
        fval2 = (*fcn)(&mut_D);

        fv1[jtw] = fval1;
        fv2[jtw] = fval2;
        fsum = fval1 + fval2;
        resg = resg + wg[j] * fsum;
        resk = resk + wgk[jtw] * fsum;
        *resabs = *resabs + wgk[jtw] * (fabs(fval1) + fabs(fval2));
    }
    // 10

    for (j = 0; j < 5; j++) {
        jtwm1 = 2 * j;
        absc = hlgth * xgk[jtwm1];

        mut_D = centr - absc;
        fval1 = (*fcn)(&mut_D);
        mut_D = centr + absc;
        fval2 = (*fcn)(&mut_D);

        fv1[jtwm1] = fval1;
        fv2[jtwm1] = fval2;
        fsum = fval1 + fval2;
        resk = resk + wgk[jtwm1] * fsum;
        *resabs = *resabs + wgk[jtwm1] * (fabs(fval1) + fabs(fval2));
    }
    // 15

    reskh = resk * 0.5;
    *resasc = wgk[10]*fabs(fc - reskh);
    for (j = 0; j < 10; j++ )
    {
        *resasc = *resasc + wgk[j] * (fabs(fv1[j] - reskh) + fabs(fv2[j] - reskh));
    }
    // 20

    *result = resk * hlgth;
    *resabs = (*resabs)*dhlgth;
    *resasc = (*resasc)*dhlgth;
    *abserr = fabs((resk - resg) * hlgth);

    if ((*resasc != 0.0) && (*abserr != 0.0))
    {
        *abserr = (*resasc) * fmin(1.0, pow((200.0 * (*abserr)/(*resasc)), 1.5));
    }
    if (*resabs > uflow/(50.0 * epmach))
    {
        *abserr = fmax(epmach * 50.0 * (*resabs),(*abserr));
    }

    return;
}

void
dqpsrt(const int limit, const int last, int* maxerr, double* ermax,
       const double* elist, int *iord, int* nrmax)
{
    // ***begin prologue  dqpsrt
    // ***refer to  dqage,dqagie,dqagpe,dqawse
    // ***routines called  (none)
    // ***revision date  810101   (yymmdd)
    // ***keywords  sequential sorting
    // ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
    //            de doncker,elise,appl. math. & progr. div. - k.u.leuven
    // ***purpose  this routine maintains the descending ordering in the
    //             list of the local error estimated resulting from the
    //             interval subdivision process. at each call two error
    //             estimates are inserted using the sequential search
    //             method, top-down for the largest error estimate and
    //             bottom-up for the smallest error estimate.
    // ***description
    //
    //            ordering routine
    //            standard fortran subroutine
    //            double precision version
    //
    //            parameters (meaning at output)
    //               limit  - integer
    //                        maximum number of error estimates the list
    //                        can contain
    //
    //               last   - integer
    //                        number of error estimates currently in the list
    //
    //               maxerr - integer
    //                        maxerr points to the nrmax-th largest error
    //                        estimate currently in the list
    //
    //               ermax  - double precision
    //                        nrmax-th largest error estimate
    //                        ermax = elist(maxerr)
    //
    //               elist  - double precision
    //                        vector of dimension last containing
    //                        the error estimates
    //
    //               iord   - integer
    //                        vector of dimension last, the first k elements
    //                        of which contain pointers to the error
    //                        estimates, such that
    //                        elist(iord(1)),...,  elist(iord(k))
    //                        form a decreasing sequence, with
    //                        k = last if last.le.(limit/2+2), and
    //                        k = limit+1-last otherwise
    //
    //               nrmax  - integer
    //                        maxerr = iord(nrmax)
    //
    // ***end prologue  dqpsrt
    //
    int i, ibeg, j, jbnd, jupbn, k;
    double errmin, errmax;
    // Index vars maxerr, iord, nrmax

    if (last <= 2)
    {
        iord[0] = 0;
        iord[1] = 1;
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    }

    // This part of the routine is only executed if, due to a difficult integrand,
    // subdivision increased the error estimate. In the normal case the insert
    // procedure should start after the nrmax-th largest error estimate.
    errmax = elist[*maxerr];
    while ((*nrmax > 0) && (errmax > elist[iord[*nrmax-1]]))
    {
        iord[*nrmax] = iord[*nrmax - 1];
        *nrmax -= 1;
    }

    // Compute the number of elements in the list to be maintained
    // in descending order. this number depends on the number of
    // subdivisions still allowed.
    jupbn = (last > limit/2 + 2 ? limit - last + 2 : last - 1);
    errmin = elist[last - 1];

    // insert errmax by traversing the list top-down,
    // starting comparison from the element elist(iord(nrmax+1)).
    jbnd = jupbn-1;
    ibeg = *nrmax + 1;

    if (ibeg <= jbnd)
    {
        for (i = ibeg; i <= jbnd; i++)
        {
            if (errmax >= elist[iord[i]]) { goto LINE60; }
            iord[i-1] = iord[i];
        }
        // 40
    }
    // 50
    iord[jbnd] = *maxerr;
    iord[jupbn] = last-1;
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
    return;
LINE60:
    // 60
    // Insert errmin by traversing the list bottom-up
    iord[i-1] = *maxerr;
    k = jbnd;
    for (j = i; j <= jbnd; j++)
    {
        if (errmin < elist[iord[k]]) { break; }
        iord[k+1] = iord[k];
        k--;
    }
    //70

    if (errmin < elist[iord[k]])
    {
        iord[k+1] = last-1;
    } else {
        iord[i] = last-1;
    }

    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];

    return;
}
