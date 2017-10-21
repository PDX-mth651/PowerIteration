#include "linalgcpp.hpp"

using namespace linalgcpp;

double PowerIterate(const Operator& op)
{
    assert(op.Rows() == op.Cols());

    const size_t size = op.Rows();

    Vector<double> vect(size);
    Vector<double> vect_next(size);

    Randomize(vect);

    constexpr double tol = 1e-6;
    constexpr int max_iter = 1000;
    constexpr bool verbose = true;

    double ray_q = 1.0;
    double ray_q_old = 1.0;

    int iter = 0;

    while (iter < max_iter)
    {
        op.Mult(vect, vect_next);

        ray_q = (vect * vect_next) / (vect * vect);

        Normalize(vect_next);

        Swap(vect, vect_next);

        const double rate = std::fabs(1.0 - (ray_q / ray_q_old));

        if (verbose)
        {

            printf("%.2d: ray_q: %.8f rate: %.2e\n",
                   iter, ray_q, rate);
        }

        iter++;

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

    SparseMatrix<int> sparse = coo.ToSparse();
    DenseMatrix dense = coo.ToDense();

    dense.Print("Input:");

    printf("Computing Dense Eval:\n");
    double max_eval_dense = PowerIterate(dense);
    printf("Max Eval Dense: %.2f\n", max_eval_dense);

    printf("\nComputing Sparse Eval:\n");
    double max_eval_sparse = PowerIterate(sparse);
    printf("Max Eval Sparse: %.2f\n", max_eval_sparse);

    printf("\nComputing Coo Eval:\n");
    double max_eval_coo = PowerIterate(coo);
    printf("Max Eval Coo: %.2f\n", max_eval_coo);

    return EXIT_SUCCESS;
}
