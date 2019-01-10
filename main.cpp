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
    unsigned long int batch_size, centroid_size;
    double *d_buf, *c_buf;
    MPI_Init(&argc, &argv);
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise with the CLI arguments
    int n_centroids = std::stoi(argv[2]); //number of centroids for the problem

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
            std::cout << "Unexpected number of arguments provided" <<std::endl;
            std::cout << help << std::endl;
            MPI_Abort(MPI_COMM_WORLD, argc);
            exit(argc);
        }
    }
    Data d(argv[1]);
    d.init_centroids(n_centroids);
    unsigned long int n = d.get_size();
    unsigned long int dim = d.get_dim();
    batch_size = (n / nodes + 1) * dim;
    centroid_size = n_centroids * dim;
    d_buf = new double[batch_size];
    c_buf = new double[centroid_size];

    int err = MPI_Bcast(
            d.centroids(),
            centroid_size,
            MPI_DOUBLE,
            ROOT,
            MPI_COMM_WORLD);

    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to send centroids!" << std::endl;
        std::cerr << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }

    std::cout << rank << ": Receiving the centroids." << std::endl;
    err = MPI_Recv(
            c_buf,
            centroid_size,
            MPI_DOUBLE,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &stat);
    if (err != MPI_SUCCESS) {
        std::cerr << "Unable to receive the centroids." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(err);
    }



    delete[] d_buf;
    delete[] c_buf;

    MPI_Finalize();
    return 0;
}
