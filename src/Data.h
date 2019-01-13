//
// Created by andrea on 31/12/18.
//

#ifndef DATA
#define DATA

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sstream>

class Data {
protected:
    // input data coords
    std::vector<double> x;

    // centroids coordinates
    std::vector<double> y;

    // size of the problem
    unsigned int ndat, ndim;
    unsigned long size;

public:
    Data(const std::string filename) {
        std::cout << "Reading file: " << filename << std::endl;

        // read input dataset into file stream
        std::fstream instream(filename, std::fstream::in);
        std::string line;
        ndim = 0;
        ndat = 0;

        if (instream) {
            while (std::getline(instream, line)) {
                std::istringstream s(line);
                double tmp;
                while (s >> tmp) {
                    if (ndat == 0)
                        ndim++;
                    x.push_back(tmp);
                }
                ndat++;
            }
            std::cout << "File successfully loaded!" << std::endl;
        }
        instream.close();
        size = x.size();
        std::cout << "The dataset is in R^" << ndim << std::endl;
        std::cout << ndat << " data points loaded." << std::endl;
        std::cout << "The total size of the problem is: " << size << std::endl;
    }

    // generate the centroids
    void init_centroids(int n) {
        std::random_device rd;
        std::mt19937_64 eng(rd());
        std::uniform_int_distribution<> dist(0, ndat - 1);
        std::cout << "Selecting " << n << " random centroids from list:" <<std::endl;
        for (int i = 0; i < n; ++i) {
            int k = dist(eng);
            for (int j = 0; j < ndim; ++j) {
                std::cout << x[k * ndim + j] << "\t";
                y.push_back(x[k * ndim + j]);
            }
            std::cout << std::endl;
        }
    }

    unsigned int get_size() {
        return ndat;
    }

    unsigned int get_dim() {
        return ndim;
    }

    // write the data into an MPI_Scatter compatible buffer
    double *dataset() {
        return x.data();
    }

    double *centroids() {
        return y.data();
    }

    // print the data to stdout
    void print() {
        for (int i = 0; i < ndat; ++i){
            for (int j = 0; j < ndim; ++j)
                std::cout << x[i * ndim + j] << "\t";
            std::cout << std::endl;
        }
    }

};

#endif
