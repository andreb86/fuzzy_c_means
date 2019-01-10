//
// Created by andrea on 31/12/18.
//

#ifndef CENTROIDS
#define CENTROIDS

#include <algorithm>
#include <fstream>
#include <string>
#include <omp.h>
#include <vector>

class Centroids {
protected:
    std::vector<double> x;
    std::vector<double> y;
    int n_centroids;

public:
    Centroids (const int n) {
        x.resize(n);
        y.resize(n);
        n_centroids = n;
        std::generate(x.begin(), x.end(), std::rand);
        std::generate(y.begin(), y.end(), std::rand);
    }
    void update(std::vector<double> new_x, std::vector<double> new_y) {
        for (int i = 0; i < n_centroids; ++i) {
            x[i] = new_x[i];
            y[i] = new_y[i];
        }
    }
    void write (std::string output) {
        std::fstream outfile(output, std::fstream::out);
        if (outfile) {
            for (int i = 0; i < n_centroids; ++i)
                outfile << x[i] << " " << y[i] << std::endl;
        }
        outfile.close();
    }
};


#endif
