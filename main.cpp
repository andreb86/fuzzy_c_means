#include "src/CMeans.h"
#include "src/Data.h"
#include "mpich/mpi.h"
#include <iostream>
#include <cstring>

#define ROOT 0
#define CACHELINE 64


int main(int argc, char **argv) {
    std::string help(
"cmeans <input file> <centroids> <fuzzyfication> <tolerance>"
"\n\nwhere:"
"\n<input file> is the input file containing the dataset"
"\n<centroids> is the number of centroids which the dataset is divided into"
"\n<fuzzyfication> fuzzyfication parameter"
"\n<tolerance> maximum allowable error on the membership vectors");
    // Initialise MPI session and relevant variables
    int nodes, rank, c_tag = 1, d_tag = 2;
    unsigned int m, n, c, b, r; // feature space, n datapoints, c centroids, batch size and remainder
    float e, f; // tolerance for the convergence of the problem and fuzzification parameter
    int *sendcounts, *displs; // arguments for the scatterv function
    double *X;
    MPI_Init(&argc, &argv);
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise the array of sizes and the array of the offsets from beginning
    sendcounts = (int *)malloc(nodes * sizeof(int));
    displs     = (int *)malloc(nodes * sizeof(int));

    // Initialise with the CLI arguments
    c = static_cast<unsigned int>(std::stoi(argv[2])); // number of centroids for the problem
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
        if (argc != 5) {
            std::cerr << "Unexpected number of arguments provided" << std::endl;
            std::cout << help << std::endl;
            MPI_Abort(MPI_COMM_WORLD, argc);
            exit(argc);
        }
        // read the data from the root process
        Data d(argv[1]);
        n = d.get_size();
        m = d.get_dim();
        X = d.dataset();

        // Calculate the dimension of the buffer that will hold the scattered dataset
        b = n / nodes;
        r = n % nodes;
        for (int i = 0; i < nodes; ++i) {
            if (i < r) {
                sendcounts[i] = (b + 1) * m;
            } else {
                sendcounts[i] = b * m;
            }
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
//            std::cout << i << ": " << sendcounts[i] << ": " << displs[i] << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
//    std::cout << rank << ": " << c << std::endl;

    int err = MPI_Bcast(&m, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send space dimension!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

//    std::cout << rank << ": " << m << std::endl;

    err = MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send problem size!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    // Scatter the counts of the datasets
    err = MPI_Scatter(sendcounts, 1, MPI_UNSIGNED, &b, 1, MPI_UNSIGNED, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to scatter batch sizes!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }
//    for (int i = 0; i < nodes; ++i) {
//        if (i == rank) {
//            std::cout << "°°°" << std::endl << rank << ": " << b << std::endl;
//            MPI_Barrier(MPI_COMM_WORLD);
//        }
//    }

    CMeans fcm = CMeans(b, m, c, f, e);

    if (rank == ROOT){
        err = MPI_Scatterv(X, sendcounts, displs, MPI_DOUBLE, fcm.x, b, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        if (err != MPI_SUCCESS) {
            std::cerr << "Unable to scatter data to processes!" << std::endl;
            std::cerr << err << std::endl;
            MPI_Abort(MPI_COMM_WORLD, err);
            exit(err);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < nodes; ++i) {
        if (i == rank) {
            std::cout << rank << ": " << fcm.x[0] << ", " << fcm.x[1] << std::endl << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
