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
    const int b, c, m, nthreads=2; // number of datapoints, number of centroids, space dimension and number of threads
//    int nthreads = omp_get_num_threads(); // number of threads
    const double f, tol; // fuzzification parameter and tolerance
    double err;

    // points, centroids coordinates, distances and the membership vector
    double *x, *y, *d, *d_sum, *u_old, *u_new, *u_pow, *u_sum;

    // initialise the value of the batch size b, number of centroids c and dimension of the problem
    CMeans(const unsigned bb,
            const unsigned mm,
            const unsigned cc,
            const double ff,
            const double ee):
    b(bb), c(cc), m(mm), f(ff), tol(ee), err(0) {

        // Allocate aligned memory

        std::printf("\nAllocating: %d points\n", b);
        x     = static_cast<double *>(_mm_malloc(sizeof(double) * bb * mm, CACHELINE));
        y     = static_cast<double *>(_mm_malloc(sizeof(double) * cc * mm, CACHELINE));
        d     = static_cast<double *>(_mm_malloc(sizeof(double) * bb * cc, CACHELINE));
        d_sum = static_cast<double *>(_mm_malloc(sizeof(double) * bb *  1, CACHELINE));
        u_old = static_cast<double *>(_mm_malloc(sizeof(double) * cc * bb, CACHELINE));
        u_new = static_cast<double *>(_mm_malloc(sizeof(double) * cc * bb, CACHELINE));
        u_pow = static_cast<double *>(_mm_malloc(sizeof(double) * cc * bb, CACHELINE));
        u_sum = static_cast<double *>(_mm_malloc(sizeof(double) * cc *  1, CACHELINE));

        if (x == nullptr || y == nullptr || d == nullptr || u_old == nullptr || u_new == nullptr || u_sum == nullptr) {
            std::perror("Unable to allocate arrays\n");
            _mm_free(x);
            _mm_free(y);
            _mm_free(d);
            _mm_free(d_sum);
            _mm_free(u_old);
            _mm_free(u_new);
            _mm_free(u_pow);
            _mm_free(u_sum);
            exit(1);
        }
    }

    // calculate the square of the distances
    void distances(const unsigned int block_size) {
        double tmp[m] __attribute__((aligned (CACHELINE))), tmp_d;
#pragma omp parallel for schedule(static) private(tmp, tmp_d) num_threads(nthreads)
        for (int ii = 0; ii < b; ii += block_size) {
            int ib = ii + block_size;
            for (int i = ii; i < std::min(ib, b); ++i) {
                d_sum[i] = 0;
                for (int j = 0; j < c; ++j) {
                    tmp_d = 0;
#pragma omp simd
                    for (int k = 0; k < m; ++k)
                        tmp[k] = std::pow(x[i * m + k] - y[j * m + k], 2);
                    for (int kk = 0; kk < m; ++kk)
                        tmp_d += tmp[kk];
                    if (tmp_d == 0)
                        d[i * c + j] = 0;
                    else
                        d[i * c + j] = std::pow(tmp_d, 1 / (1 - f));
                    d_sum[i] += d[i * c + j];
                }
            }
        }
    }

    // calculate the weights of each data point as ownership of a certain centroid
    void weights(const unsigned int block) {
        double tmp;
#pragma omp parallel for schedule(static) private(tmp) num_threads(nthreads)
        for (int ii = 0; ii < b; ii += block) {
            int ib = ii + block;
            for (int j = 0; j < c; ++j) {
                tmp = 0;
                for (int i = ii; i < std::min(ib, b); ++i) {
                    u_new[j * b + i] = d[i * c + j] / d_sum[i];
                    if (u_new[j * b + i] == 0)
                        u_pow[j * b + i] = 0;
                    else
                        u_pow[j * b + i] = std::pow(u_new[j * b + i], f);
                    tmp += u_pow[j * b + i];
                }
                u_sum[j] = tmp;
            }
        }
    }

    void check(const unsigned int block) {
        double tmp;
#pragma omp parallel for schedule(static) private(tmp) num_threads(nthreads) shared(err)
        for (int ii = 0; ii < b; ii += block) {
            int ib = ii + block;
            for (int j = 0; j < c; ++j) {
                for (int i = ii; i < std::min(ib, b); ++i) {
                    tmp = std::abs(u_new[j * b + i] - u_old[j * b + i]);
                    if (tmp > err)
#pragma omp atomic write
                        err = tmp;
                }
            }
        }
        if (err > tol) {
            // Swap the old and new values of the membership vectors
            std::swap(u_old, u_new);
        }
    }

    void umulx(const unsigned int block) {
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int ii = 0; ii < c; ii += block) {
            int ib = ii + block;
            for (int kk = 0; kk < b; kk += block) {
                int kb = kk + block;
                for (int jj = 0; jj < m; jj += block) {
                    int jb = jj + block;
                    for (int i = ii; i < std::min(ib, c); ++i)
                        for (int k = kk; k < std::min(kb, b); ++k)
                            for (int j = jj; j < std::min(jb, m); ++j) {
                                if (!k) y[i * m + j] = 0;
                                y[i * m + j] += u_pow[i * b + k] * x[k * m + j];
                            }
                }
            }
        }
    }

    ~CMeans() {
        std::cout.flush();
        std::printf("Freeing up memory...");
        _mm_free(x);
        _mm_free(y);
        _mm_free(d);
        _mm_free(d_sum);
        _mm_free(u_old);
        _mm_free(u_new);
        _mm_free(u_pow);
        _mm_free(u_sum);
        x     = nullptr;
        y     = nullptr;
        d     = nullptr;
        d_sum = nullptr;
        u_old = nullptr;
        u_new = nullptr;
        u_pow = nullptr;
        u_sum = nullptr;

        std::cout.flush();
        std::cout << "Bye!" << std::endl;
    }
};

#endif
