#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "batman.h"

#define DEC2RAD(x) ((x)*M_PI/180.0)

double *my_alloc(int length) {
    double *data = malloc(length * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Error allocating memory\n");
        exit(1);
    }
    return data;
}

// XXX data must be of the correct length
void linspace(double *data, double min, double max, int N) {
    int i;
    double step = (max - min) / (double)(N - 1);
    for (i=0; i<N; i++) {
        data[i] = min + i * step;
    }
}

double get_fac(const Params *params, LimbDarkeningType ldc_type) {
    int i, nds = 1000;
    double fac_low = 5.0E-4, fac_high = 1.0;
    double *ds = my_alloc(nds);
    double *f0 = my_alloc(nds);
    double *f = my_alloc(nds);

    linspace(ds, 0., 1. + params->rp, nds);

    /* Initialise */
    switch (ldc_type) {
        case NONLINEAR:
            nonlinear_ld(ds, f0, nds, params->rp,
                    params->nlldc.c1, params->nlldc.c2, params->nlldc.c3, params->nlldc.c4, fac_low, 1);
            break;
        case EXPONENTIAL:
            exponential_ld(ds, f0, nds, params->rp, params->ldc.c1, params->ldc.c2, fac_low, 1);
            break;
        case LOGARITHMIC:
            logarithmic_ld(ds, f0, nds, params->rp, params->ldc.c1, params->ldc.c2, fac_low, 1);
            break;
        default:
            fprintf(stderr, "Non-supported ldc type\n");
            exit(1);
            break;
    }

    double err = 0.;
    int n = 0;
    double max_error = 1.0;
    double fac = 0;

    while ((err > max_error) || (err < (0.99 * max_error))) {
        fac = (fac_low + fac_high) / 2.0;
        switch (ldc_type) {
            case NONLINEAR:
                nonlinear_ld(ds, f, nds, params->rp,
                    params->nlldc.c1, params->nlldc.c2, params->nlldc.c3, params->nlldc.c4, fac, 1);
                break;
            case EXPONENTIAL:
                exponential_ld(ds, f, nds, params->rp, params->ldc.c1, params->ldc.c2, fac, 1);
                break;
            case LOGARITHMIC:
                logarithmic_ld(ds, f, nds, params->rp, params->ldc.c1, params->ldc.c2, fac, 1);
                break;
            default:
                fprintf(stderr, "Non-supported ldc type\n");
                exit(1);
                break;
        }

        double max_abs_derror = f[0] - f0[0];
        for (i=0; i<nds; i++) {
            double derror = fabs(f[i] - f0[i]);
            if (derror > max_abs_derror) {
                max_abs_derror = derror;
            }
        }
        err = max_abs_derror * 1E6;

        if (err > max_error) {
            fac_high = fac;
        } else {
            fac_low = fac;
        }

        n += 1;
        if (n > 1E3) {
            fprintf(stderr, "Convergence failure in calculation of scale factor for integration step size\n");
            exit(1);
        }
    }


    free(f);
    free(f0);
    free(ds);
    return fac;
}

double *light_curve(const Params *params, const double *t, const int length, LimbDarkeningType limb_darkening_type) {
    int nthreads = 1;
    double *flux = my_alloc(length);
    double *ds = my_alloc(length);

    rsky(t, ds, length, params->t0, params->per, params->a, DEC2RAD(params->inc), params->ecc, DEC2RAD(params->w), 1);

    double fac = 0.;


    switch (limb_darkening_type) {
        case UNIFORM:
            uniform_ld(ds, flux, length, params->rp, nthreads);
            break;
        case LINEAR:
            quadratic_ld(ds, flux, length, params->rp, params->ldc.c1, 0., nthreads);
            break;
        case EXPONENTIAL:
            fac = get_fac(params, limb_darkening_type);
            exponential_ld(ds, flux, length, params->rp, params->ldc.c1, params->ldc.c2, fac, nthreads);
            break;
        case LOGARITHMIC:
            fac = get_fac(params, limb_darkening_type);
            logarithmic_ld(ds, flux, length, params->rp, params->ldc.c1, params->ldc.c2, fac, nthreads);
            break;
        case QUADRATIC:
            quadratic_ld(ds, flux, length, params->rp, params->ldc.c1, params->ldc.c2, nthreads);
            break;
        case NONLINEAR:
            fac = get_fac(params, limb_darkening_type);
            nonlinear_ld(ds, flux, length, params->rp, params->nlldc.c1, params->nlldc.c2, params->nlldc.c3, params->nlldc.c4,
                fac, nthreads);
            break;
        case SQUAREROOT:
            fac = get_fac(params, limb_darkening_type);
            nonlinear_ld(ds, flux, length, params->rp, params->ldc.c2, params->ldc.c1, 0., 0., fac, nthreads);
            break;
        default:
            fprintf(stderr, "Unsupported limb darkening type\n");
            exit(1);
            break;
    }

    free(ds);
    return flux;
}
