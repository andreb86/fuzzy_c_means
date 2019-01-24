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
"\n<tolerance> maximum allowable error on the membership vectors"
"\n<block> block size");
    // Initialise MPI session and relevant variables
    int nodes, rank;
    unsigned int m, n, c, b, r, block; // feature space, n datapoints, c centroids, batch size, remainder and block
    float e, f; // tolerance for the convergence of the problem and fuzzification parameter
    int *sendcounts, *displs; // arguments for the scatterv function
    double *X;

    // TEST BUFFER
    int recv_buf[100];

    Data *d;
    MPI_Init(&argc, &argv);
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise the array of sizes and the array of the offsets from beginning
    sendcounts = (int *)malloc(nodes * sizeof(int));
    displs     = (int *)malloc(nodes * sizeof(int));

    // Initialise with the CLI arguments
    c = static_cast<unsigned int>(std::stoi(argv[2])); // number of centroids for the problem
    block = static_cast<unsigned int>(std::stoi(argv[5])); // block size
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
        d = new Data(argv[1]);
        n = d->get_size();
        m = d->get_dim();
//        X = d->dataset();

        // Calculate the dimension of the buffer that will hold the scattered dataset
        b = n / nodes;
        r = n % nodes;
        std::cout << "================= P A R T I T I O N I N G ====================" << std::endl;
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
            std::cout << "Process: " << i << " N. elements: " << sendcounts[i] << " Offset: " << displs[i] << std::endl;
        }
    }

    int err = MPI_Bcast(&m, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send space dimension!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    err = MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send problem size!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    // Calculate the number of data points to be sent to each process
    b = n / nodes;
    r = n % nodes;
    if (rank < r) {
        b++;
    }

    // Create the clustering class for each process
    CMeans fcm = CMeans(b, m, c, f, e);
    std::cout.flush();
    err = MPI_Scatterv(d->dataset(), sendcounts, displs, MPI_DOUBLE, fcm.x, b * m, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr.flush();
        std::cerr << "Unable to scatter data to processes!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    do {
        fcm.distances()
    }



    // Cleanup;
    if (rank == ROOT)
        delete [] d;
    MPI_Finalize();
    return 0;
}
