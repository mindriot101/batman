#ifndef BATMAN_H_
#define BATMAN_H_

// _rsky.c
double getE(double M, double e);
void rsky(double *ts, double *ds, int len, double tc, double per, double a, double inc, double ecc, double omega, int transittype);
void getf(double *ts, double *fs, int len, double tc, double per, double a, double inc, double ecc, double omega, int transittype);

// _eclipse.c
void eclipse(double *ds, double *fs, int len, double p, double fp, int nthreads);

// _nonlinear.c
double intensity(double x, double c1, double c2, double c3, double c4, double norm);
double area(double d, double x, double R);
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

double *light_curve(Params *params, double *t, int length);

#endif //  BATMAN_H_
