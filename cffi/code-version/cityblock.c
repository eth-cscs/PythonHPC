#include <math.h>


void cbdm(double *a, double *b, double *r, int num_rows, int num_cols) {
    double _r;

#pragma omp parallel for reduction (+:_r)
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_rows ; j++) {
            _r = 0.0;
            for(int k = 0; k < num_cols ; k++) {
                _r += fabs(a[i * num_cols + k] - b[j * num_cols + k]);
            }
            r[i * num_rows + j] = _r;
        }
    }
}
