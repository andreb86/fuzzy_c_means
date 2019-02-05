#include "src/CMeans.h"
#include "src/Data.h"
#include "mpich/mpi.h"
#include <iostream>
#include <mm_malloc.h>
#include <cstring>

#define ROOT 0
#define CACHELINE 64


int main(int argc, char **argv) {
    std::string help(
"cmeans <input file> <centroids> <fuzzyfication> <tolerance> <block>"
"\n\nwhere:"
"\n<input file>\tis the input file containing the dataset"
"\n<centroids>\tis the number of centroids which the dataset is divided into"
"\n<fuzzyfication>\tfuzzyfication parameter"
"\n<tolerance>\tmaximum allowable error on the membership vectors"
"\n<block>\tblock size");
    // Initialise MPI session and relevant variables
    int nodes, rank;
    unsigned int m, n, c, b, r, block; // feature space, n datapoints, c centroids, batch size, remainder and block
    double e, f; // tolerance for the convergence of the problem and fuzzification parameter
    int *counts, *displs; // arguments for the scatterv function

    Data *d;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double *y, *usum;

    // Initialise the array of sizes and the array of the offsets from beginning
    counts = (int *)malloc(nodes * sizeof(int));
    displs = (int *)malloc(nodes * sizeof(int));

    // Initialise with the CLI arguments
    c = static_cast<unsigned int>(std::stoi(argv[2])); // number of centroids for the problem
    block = static_cast<const unsigned int>(std::stoi(argv[5])); // block size
    f = std::stof(argv[3]); // fuzzification parameter
    e = std::stof(argv[4]); // tolerance

    MPI_Barrier(MPI_COMM_WORLD);

    // Load data points in the root process
    if (rank == ROOT) {
        std::cout << "Found " << nodes << " nodes" << std::endl;

        // Show the help if required
        if (strcmp(argv[1], "--help") == 0) {
            std::cout << help << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 0);
            exit(0);
        }

        // Handle incorrect input
        if (argc != 6) {
            std::cerr << "Unexpected number of arguments provided" << std::endl;
            std::cout << help << std::endl;
            MPI_Abort(MPI_COMM_WORLD, argc);
            exit(argc);
        }
        // read the data from the root process
        d = new Data(argv[1]);
        n = d->get_size();
        m = d->get_dim();
        d->init_centroids(c);

        // Calculate the dimension of the buffer that will hold the scattered dataset
        b = n / nodes;
        r = n % nodes;
        std::cout << std::endl << "================= P A R T I T I O N I N G ====================" << std::endl;
        for (int i = 0; i < nodes; ++i) {
            if (i < r) {
                counts[i] = (b + 1) * m;
            } else {
                counts[i] = b * m;
            }

            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i - 1] + counts[i - 1];
            }
            std::printf("Process: %d N. elements: %d Offset: %d\n", i, counts[i], displs[i]);
        }
    }

    // Broadcast the feature space dimension
    int err = MPI_Bcast(&m, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send space dimension!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    // Initialise the vector of the centroids and the u sums
    y    = static_cast<double *>(_mm_malloc(sizeof(double) * c * m, CACHELINE));
    usum = static_cast<double *>(_mm_malloc(sizeof(double) * c * 1, CACHELINE));
    if (rank == ROOT) {
        std::memcpy(y, d->centroids(), c * m);
    }

    // Broadcast the dataset size
    err = MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send problem size!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    // Broadcast the random centroids
    err = MPI_Bcast(y, c * m, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::printf("Unable to send centroids: %d\n", err);
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    // Test print
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate the number of data points to be sent to each process
    b = n / nodes;
    r = n % nodes;
    if (rank < r) {
        b++;
    }

    // Create the clustering class for each process
    CMeans fcm = CMeans(b, m, c, f, e);
    std::cout.flush();
    if (rank == ROOT) {
        err = MPI_Scatterv(d->dataset(), counts, displs, MPI_DOUBLE, fcm.x, b * m, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    } else {
        err = MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, fcm.x, b * m, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }
    if (err != MPI_SUCCESS) {
        std::printf("Unable to send data batch: %d\n", err);
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    std::printf("Process No. %d -> Beginning of dataset (%f, %f)\n", rank, fcm.x[0], fcm.x[1]);

    // Copy the centroids into the aligned buffer
    std::memcpy(fcm.y, y, c * m * sizeof(double));
    double diff = 100;
    int counter = 0;

    fcm.distances(block);
    fcm.weights(block);
    std::cout.flush();
    std::printf("\nDistance of the first point: %e", fcm.d[0]);

//    do {
//        fcm.distances(block);
//        fcm.weights(block);
//        fcm.check(block);

        // Find the maximum error among all of the processes
//        err = MPI_Allreduce(&fcm.err, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//        if (err != MPI_SUCCESS){
//            std::printf("Error while reducing the error: %d", err);
//            MPI_Abort(MPI_COMM_WORLD, err);
//            exit(err);
//        }
//        fcm.umulx(block);

        // Reduce the centroids
//        err = MPI_Allreduce(fcm.y, y, c * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        if (err != MPI_SUCCESS) {
//            std::printf("Error while reducing the centroids: %d", err);
//            MPI_Abort(MPI_COMM_WORLD, err);
//            exit(err);
//        }

        // Reduce the sum of u
//        err = MPI_Allreduce(fcm.u_sum, usum, c, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        if (err != MPI_SUCCESS) {
//            std::printf("Error while reducing the sums of u: %d", err);
//            MPI_Abort(MPI_COMM_WORLD, err);
//            exit(err);
//        }

        // Calculate the updated centroids and copy them back into
//#pragma omp parallel for num_threads(c)
//        for (int i = 0; i < c; ++i) {
//            for (int j = 0; j < m; ++j) {
//                y[i * m + j] /= usum[i];
//            }
//        }
//        std::memcpy(fcm.y, y, c * m * sizeof(double));
//        counter++;
//    } while (diff > e & counter < 1);

    if (rank == ROOT) {
        std::cout.flush();
        std::cout << std::endl << "================= R E S U L T S ====================" << std::endl;
        for (int i = 0; i < c; ++i) {
            for (int k = 0; k < m; ++k) {
                if(k % m == 0) std::cout << std::endl;
                std::printf("%.8f,\t", fcm.y[i * m + k]);
            }
        }
    }

    // Cleanup;
    if (rank == ROOT)
        delete d;
    _mm_free(y);
    _mm_free(usum);
    MPI_Finalize();
    return 0;
}
