#include "src/CMeans.h"
#include "src/Data.h"
#include "mpich/mpi.h"
#include <iostream>
#include <cstring>

#define ROOT 0
#define CACHELINE 64


int main(int argc, char **argv) {
    std::string help(
"cmeans <input file> <clusters> <processes> <cores>"
"\n\nwhere:"
"\n<input file> is the input file containing the dataset"
"\n<centroids> is the number of centroids which the dataset is divided into");
    // Initialise MPI session and relevant variables
    int nodes, rank, c_tag = 1, d_tag = 2;
    unsigned int m, n, c, b; // dimension of the problem, number of datapoints, number of centroids and batch
    double *d_buf = nullptr, *c_buf = nullptr; // TODO add a buffer to receive scattered data points
    MPI_Init(&argc, &argv);
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise the buffer to store datapoints with alignment to the cacheline

    // Initialise with the CLI arguments
    c = std::stoi(argv[2]); //number of centroids for the problem

    MPI_Barrier(MPI_COMM_WORLD);
    // Load data points in the root process
    if (rank == ROOT) {
        std::cout << "Found " << nodes << " nodes" << std::endl;

        if (strcmp(argv[1], "--help") == 0) {
            std::cout << help << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 0);
            exit(0);
        }

        if (argc != 3) {
            std::cerr << "Unexpected number of arguments provided" << std::endl;
            std::cout << help << std::endl;
            MPI_Abort(MPI_COMM_WORLD, argc);
            exit(argc);
        }
        // read the data from the root process
        Data d(argv[1]);
        d.init_centroids(c);
        n = d.get_size();
        m = d.get_dim();
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


//    MPI_Scatter(, );

    MPI_Finalize();
    return 0;
}
