#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "batman.h"

static double *my_alloc(int length) {
    double *data = malloc(length * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Error allocating memory\n");
        exit(1);
    }
    return data;
}

void read_hjd(const char *filename, double **hjd, int *length) {
    FILE *f = fopen(filename, "r");
    char buf[256];
    if (f == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    /* get the length of the file */
    *length = 0;
    while (fgets(buf, 256, f) != NULL) {
        *length += 1;
    }
    fseek(f, 0, SEEK_SET);

    *hjd = my_alloc(*length);

    int counter = 0;
    while (fgets(buf, 256, f) != NULL) {
        (*hjd)[counter] = atof(buf);
        counter++;
    }

    assert(counter == *length);

    fclose(f);
}

int main() {

    NonlinearLimbDarkeningParameters ldc;
    ldc.c1 = 0.7692;
    ldc.c2 = -0.716;
    ldc.c3 = 1.1874;
    ldc.c4 = -0.5372;

    Params params;
    params.t0 = 0.;
    params.per = 1.;
    params.rp = 0.1;
    params.a = 15.;
    params.inc = 87;
    params.ecc = 0;
    params.w = 90;
    params.ldc = ldc;

    double *hjd = NULL;
    int length = 0, i = 0;

    read_hjd("testtimes.txt", &hjd, &length);
    printf("%d times read\n", length);

    double *lc = light_curve(&params, hjd, length);

    FILE *outfile = fopen("c-lightcurve.txt", "w");
    for (i=0; i<length; i++) {
        fprintf(outfile, "%lf %lf\n", hjd[i], lc[i]);
    }
    fclose(outfile);


    free(lc);
    free(hjd);

    return 0;
}
