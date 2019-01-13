#include "src/CMeans.h"
#include "src/Data.h"
#include "mpich/mpi.h"
#include <iostream>
#include <cstring>

#define ROOT 0


int main(int argc, char **argv) {
    std::string help(
"cmeans <input file> <clusters> <processes> <cores>"
"\n\nwhere:"
"\n<input file> is the input file containing the dataset"
"\n<centroids> is the number of centroids which the dataset is divided into");
    // Initialise MPI session and relevant variables
    int nodes, rank, c_tag = 1, d_tag = 2;
    unsigned int dim, ndat, ncen; // dimension of the problem, number of datapoints and number of centroids
    double *d_buf = nullptr, *c_buf = nullptr;
    MPI_Init(&argc, &argv);
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise with the CLI arguments
    ncen = std::stoi(argv[2]); //number of centroids for the problem

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
        d.init_centroids(ncen);
        ndat = d.get_size();
        dim = d.get_dim();
        c_buf = d.centroids();
    }

//    MPI_Barrier(MPI_COMM_WORLD);
//    std::cout << rank << ": " << ncen << std::endl;

    int err = MPI_Bcast(
            &dim,
            1,
            MPI_INT,
            ROOT,
            MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send problem dimension!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

//    std::cout << rank << ": " << dim << std::endl;

    err = MPI_Bcast(
            &ndat,
            1,
            MPI_INT,
            ROOT,
            MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send problem size!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    // Broadcast the centroids to all the processes
    MPI_Bcast(c_buf, ncen * dim, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    for (int k = 0; k < ncen * dim; ++k) {
        std::cout << rank << ": " << c_buf[k] << std::endl;
    }

    int batch = ndat * dim / nodes
    MPI_Scatter(, );

    MPI_Finalize();
    return 0;
}
