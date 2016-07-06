#ifndef BATMAN_H_
#define BATMAN_H_

#ifdef __cplusplus
extern "C" {
#endif

// _rsky.c
double getE(double M, double e);
void rsky(const double *ts, double *ds, int len, double tc, double per, double a, double inc, double ecc, double omega, int transittype);
void getf(double *ts, double *fs, int len, double tc, double per, double a, double inc, double ecc, double omega, int transittype);

// _eclipse.c
void eclipse(double *ds, double *fs, int len, double p, double fp, int nthreads);

// _exponential_ld.c
void exponential_ld(double *ds, double *fs, int len, double rprs, double c1, double c2, double fac, int nthreads);

// _logarithmic_ld.c
void logarithmic_ld(double *ds, double *fs, int len, double rprs, double c1, double c2, double fac, int nthreads);

// _quadratic_ld.c
void quadratic_ld(double *ds, double *fs, int len, double p, double c1, double c2, int nthreads);

// _nonlinear.c
void nonlinear_ld(double *ds, double *fs, int len, double rprs, double c1, double c2, double c3, double c4, double fac, int nthreads);

// light_curve.c

typedef struct {
    double c1;
    double c2;
    double c3;
    double c4;
} NonlinearLimbDarkeningParameters;

typedef struct {
    double t0;
    double per;
    double rp;
    double a;
    double inc;
    double ecc;
    double w;
    union {
        NonlinearLimbDarkeningParameters ldc;
    };
} Params;

double *light_curve(const Params *params, const double *t, const int length);

#ifdef __cplusplus
}
#endif

#endif //  BATMAN_H_
