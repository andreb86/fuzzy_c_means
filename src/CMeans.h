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
#include </home/andrea/amd/blis/include/blis/cblas.h>

class CMeans {
public:
    const double f, e; // fuzzification parameter and tolerance

    // points, centroids coordinates, distances and the membership vector
    double *x, *y, *d, *u_old, *u_new, *u_sum;

    // number of datapoints, number of centroids and space dimension
    const int b, c, m;
    int nthreads = omp_get_num_threads();
    // initialise the value of the batch size b, number of centroids c and dimension of the problem
    CMeans(const unsigned bb, const unsigned mm, const unsigned cc, const double ff, const double ee):
    b(bb), c(cc), m(mm), f(ff), e(ee) {

        // Allocate aligned memory
        std::cout << "Allocating memory for data arrays." << std::endl;
        x     = static_cast<double *>(_mm_malloc(sizeof(double) * bb * mm, CACHELINE));
        y     = static_cast<double *>(_mm_malloc(sizeof(double) * cc * mm, CACHELINE));
        d     = static_cast<double *>(_mm_malloc(sizeof(double) * bb * cc, CACHELINE));
        u_old = static_cast<double *>(_mm_malloc(sizeof(double) * cc * bb, CACHELINE));
        u_new = static_cast<double *>(_mm_malloc(sizeof(double) * cc * bb, CACHELINE));
        u_sum = static_cast<double *>(_mm_malloc(sizeof(double) * cc *  1, CACHELINE));

        if (x == NULL || y == NULL || d == NULL || u_old == NULL || u_new == NULL || u_sum == NULL) {
            std::cerr << "Unable to allocate arrays." << std::endl;
            _mm_free(x);
            _mm_free(y);
            _mm_free(d);
            _mm_free(u_old);
            _mm_free(u_new);
            _mm_free(u_sum);
            exit(1);
        }

        // initialise the membership vectors
#pragma vector aligned
        for (int k = 0; k < b * c; ++k) {
            u_new[k] = u_old[k] = 1 / c;
        }
    }

    // calculate the square of the distances
    void distances(const unsigned int block_size) {
        double tmp[m] __attribute__((aligned (CACHELINE)));
#pragma omp parallel for schedule(static) private(tmp) num_threads(nthreads)
        for (int ii = 0; ii < b; ii += block_size) {
            int ib = ii + block_size;
            for (int j = 0; j < c; ++j)
                for (int i = ii; i < std::min(ib, b); ++i) {
                    d[i * c + j] = 0;
#pragma omp simd
                    for (int k = 0; k < m; ++k)
                        tmp[k] = std::pow(x[i * m + k] - y[j * m + k], 2);
                    for (int kk = 0; kk < m; ++kk)
#pragma omp atomic
                        d[i * c + j] += tmp[kk];
                }
        }
    }

    // calculate the weights of each data point as ownership of a certain centroid
    void weights(const unsigned int block) {
        double sum, d1;
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int ii = 0; ii < b; ii += block) {
            int ib = ii + block;
            for (int j = 0; j < c; ++j) {
                for (int i = ii; i < std::min(ib, b); ++i) {
                    sum = 0.0;
                    d1 = std::pow(d[i * c + j], 1 / (1 - f));
                    for (int k =  0; k < c; ++k) {
                        sum += std::pow(d[i * c + k], 1 / (1 - f));
                    }
                    u_new[j * b + i] = d1 / sum;

                }
            }
        }
    }

    void sums(const unsigned int block) {
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int ii = 0; ii < b; ii += block) {
            int ib = ii + block;
            for (int j = 0; j < c; ++j) {
                u_sum[j] = 0;
                for (int i = ii; i < std::min(ib, b); ++i) {
                    u_sum[j] += u_new[j * b + i];
                }
            }
        }
    }

    bool check(const unsigned int block) {
        double err = 0;
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int ii = 0; ii < b; ii += block) {
            int ib = ii + block;
            for (int j = 0; j < c; ++j) {
                for (int i = ii; i < std::min(ib, b); ++i) {
                    err += std::pow(u_new[j * b + i] - u_old[j * b + i], 2);
                    if (err >= e) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void update() {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, m, b, 1, u_old, b, x, m, 0, y, m);
    }

    ~CMeans() {
        std::cout << "Freeing up memory..." << std::endl;
        _mm_free(x);
        _mm_free(y);
        _mm_free(d);
        _mm_free(u_old);
        _mm_free(u_new);
        _mm_free(u_sum);
        std::cout << "Bye!" << std::endl;
    }
};

#endif
