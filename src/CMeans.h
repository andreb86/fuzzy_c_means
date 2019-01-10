//
// Created by andrea on 01/01/19.
//

#ifndef CMEANS
#define CMEANS
#define CACHELINE 64

#include "Data.h"
#include <stdlib.h>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
#include <mm_malloc.h>

class CMeans {
public:
    double f; // fuzzification parameter

    // points, centroids coordinates, distances and the membership vector
    double *x, *y, *d, *u;

    // sizes of the data and centroids
    int batch_size, centroid_size, dim;

    CMeans(int n, int m, int c, double *dbuf, double *cbuf) {

        std::cout << "Allocating memory for data buffers." << std::endl;
        // Allocate the memory
//        x = new double[n * m + CACHELINE / sizeof(double)];
//        y = new double[c * m + CACHELINE / sizeof(double)];
//        u = new double[n * c + CACHELINE / sizeof(double)];
//        d = new double[n * c + CACHELINE / sizeof(double)];


        // Align the buffers to memory
//        x = (double *)(((unsigned long) x + CACHELINE) & ~(CACHELINE - 1));
//        y = (double *)(((unsigned long) y + CACHELINE) & ~(CACHELINE - 1));
//        u = (double *)(((unsigned long) u + CACHELINE) & ~(CACHELINE - 1));
//        d = (double *)(((unsigned long) d + CACHELINE) & ~(CACHELINE - 1));

        // Allocate aligned memory
        x = static_cast<double *>(_mm_malloc(sizeof(double) * n * m, CACHELINE));
        y = static_cast<double *>(_mm_malloc(sizeof(double) * c * m, CACHELINE));
        d = static_cast<double *>(_mm_malloc(sizeof(double) * n * c, CACHELINE));
        u = static_cast<double *>(_mm_malloc(sizeof(double) * n * c, CACHELINE));


        // copy the data points
#pragma omp for schedule(static)
        for (int i = 0; i < n * m; ++i) {
            x[i] = dbuf[i];
        }

        // copy the centroids
#pragma omp for schedule(static)
        for (int j = 0; j < c * m; ++j) {
            y[j] = cbuf[j];
        }

        // initialise to 0 the distances array and the membership vectors
#pragma omp for schedule(static)
        for (int k = 0; k < n * c; ++k) {
            u[k] = d[k] = 0;
        }
        batch_size = n;
        centroid_size = c;
        dim = m;
    }

    double *distances(const unsigned int block_size) {
        double sum;
#pragma omp parallel for schedule(static) reduction(+:sum)
        for (int ii = 0; ii < batch_size; ii += block_size) {
            int ib = ii + block_size;
            for (int j = 0; j < centroid_size; ++j) {
                for (int i = ii; i < std::min(ib, batch_size); ++i) {
                    sum = 0;
#pragma omp simd
                    for (int k = 0; k < dim; ++k)
                        sum += std::pow(x[i * dim + k] - y[j * dim + k], 2);
                    d[i * centroid_size + j] = sum;
                }
            }
        }
        return d;
    }

    double *membership(const unsigned int block_size) {
        double sum, d1, d2;
#pragma omp parallel for schedule(static) reduction(+:sum)
        for (int ii = 0; ii < batch_size; ii += block_size) {
            int ib = ii + block_size;
            for (int j = 0; j < centroid_size; ++j) {
                for (int i = ii; i < std::min(ib, batch_size); ++ii) {
                    sum = 0.0;
                    d1 = std::pow(d[i * centroid_size + j], 1 / (f - 1));
#pragma omp simd
                    for (int k =  0; k < centroid_size; ++k) {
                        sum += 1 / std::pow(d[ii * centroid_size + k], 1 / (f - 1));
                    }
                    u[ii *centroid_size + j] = 1 / (d1 * sum);
                }
            }
        }
        return u;
    }

    ~CMeans() {
        std::cout << "Freeing up memory..." << std::endl;
//        delete [] x;
//        delete [] y;
//        delete [] d;
//        delete [] u;
        _mm_free(x);
        _mm_free(y);
        _mm_free(d);
        _mm_free(u);
        std::cout << "Bye!" << std::endl;
    }
};


#endif
