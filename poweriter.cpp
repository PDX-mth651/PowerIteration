/* A simple example of using linalgcpp
 */

#include "linalgcpp.hpp"

using namespace linalgcpp;

/* Performs power iterations to find the maximum eigenvalue.
 *
 * Computes until a tolerance is reached:
 *     b_k_next = A b_k
 *     b_k = b_k_next / ||b_k_next||
 *
 * Returns the computed maximum eigenvalue
 *
 */
double PowerIterate(const Operator& op, int max_iter = 1000,
                    double tol = 1e-6, bool verbose = false)
{
    assert(op.Rows() == op.Cols());

    const size_t size = op.Rows();

    // Setup Vectors and initialize with random guess
    Vector<double> vect(size);
    Vector<double> vect_next(size);

    Randomize(vect);

    // Keep track of the Rayleigh quotient
    double ray_q = 1.0;
    double ray_q_old = 1.0;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        op.Mult(vect, vect_next);

        ray_q = (vect * vect_next) / (vect * vect);

        vect_next /= L2Norm(vect_next);

        Swap(vect, vect_next);

        const double rate = std::fabs(1.0 - (ray_q / ray_q_old));

        if (verbose)
        {
            printf("%.2d: ray_q: %.8f rate: %.2e\n",
                   iter, ray_q, rate);
        }

        if (rate < tol)
        {
            break;
        }

        ray_q_old = ray_q;
    }

    return ray_q;
}

int main(int argc, char** argv)
{
    // Create some test matrix
    CooMatrix<int> coo(5, 5);

    coo.AddSym(0, 0, 2);
    coo.AddSym(0, 1, -1);
    coo.AddSym(0, 2, -1);
    coo.AddSym(1, 1, 2);
    coo.AddSym(1, 2, -1);
    coo.AddSym(2, 2, 4);
    coo.AddSym(2, 3, -1);
    coo.AddSym(2, 4, -1);
    coo.AddSym(3, 3, 2);
    coo.AddSym(3, 4, -1);
    coo.AddSym(4, 4, 2);

    // Create different operators
    SparseMatrix<int> sparse = coo.ToSparse();
    DenseMatrix dense = coo.ToDense();

    dense.Print("Input:");

    // Compute Max Eigenvalues using different operators
    constexpr int max_iter = 100;
    constexpr double tol = 1e-6;
    constexpr bool verbose = false;

    double eval_dense = PowerIterate(dense, max_iter, tol, verbose);
    double eval_sparse = PowerIterate(sparse, max_iter, tol, verbose);
    double eval_coo = PowerIterate(coo, max_iter, tol, verbose);

    std::cout << "Max Eval Dense:  " << eval_dense << "\n";
    std::cout << "Max Eval Sparse: " << eval_sparse << "\n";
    std::cout << "Max Eval Coo:    " << eval_coo << "\n";

    return EXIT_SUCCESS;
}
