#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <random>

// Tipo a virgola mobile usato
using FloatType = double;
using Scalar = std::complex<FloatType>;
using CMatrix = Eigen::MatrixXcd;
using CVector = Eigen::VectorXcd;

namespace randla::random {

// Generatore standard di vettori gaussiani complessi
struct RandomGenerator {
    static CVector randomComplexGaussianVector(int n, int seed) {
        std::mt19937 gen(seed);
        std::normal_distribution<FloatType> dist(0.0, 1.0);

        CVector vec(n);
        for (int i = 0; i < n; ++i) {
            FloatType real = dist(gen);
            FloatType imag = dist(gen);
            vec(i) = Scalar(real, imag);
        }
        return vec;
    }
};

} // namespace randla::random

// Classe che contiene la funzione da testare
struct MatrixFactorizer {
    static Scalar estimateSpectralNorm(const CMatrix& E, int seed, int power_steps = 6) {
        const int n = E.cols();
        CVector z = randla::random::RandomGenerator::randomComplexGaussianVector(n, seed);
        z.normalize();

        for (int i = 0; i < power_steps; ++i) {
            z = E.adjoint() * (E * z);
            z.normalize();
        }

        return (E * z).norm();
    }
};

int main() {
    // Crea una matrice casuale complessa (es: 10x10)
    int n = 10;
    int seed = 42;

    std::mt19937 gen(seed);
    std::normal_distribution<FloatType> dist(0.0, 1.0);

    CMatrix A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = Scalar(dist(gen), dist(gen));

    // Stima la norma spettrale
    Scalar est_norm = MatrixFactorizer::estimateSpectralNorm(A, seed);
    std::cout << "Estimated spectral norm: " << std::abs(est_norm) << std::endl;

    // Confronta con norma spettrale effettiva
    Eigen::BDCSVD<CMatrix> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double true_norm = svd.singularValues()(0);
    std::cout << "True spectral norm:      " << true_norm << std::endl;

    std::cout << "Relative error:          " << std::abs(true_norm - std::abs(est_norm)) / true_norm << std::endl;

    return 0;
}
