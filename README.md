# Randomized Linear Algebra

Randomized linear algebra is a field of numerical linear algebra that uses randomization as a computational tool to develop faster and more scalable algorithms. These algorithms are particularly useful for dealing with large-scale data and matrices.

## Key Concepts

- **Randomized Sampling**: Techniques to sample rows or columns of a matrix randomly to reduce its size while preserving its essential properties.
- **Sketching**: Creating a smaller representation (sketch) of a matrix that approximates the original matrix.
- **Low-Rank Approximation**: Finding a matrix of lower rank that approximates the original matrix, often using randomized methods.

## Applications

- **Machine Learning**: Speeding up algorithms for training models and making predictions.
- **Data Compression**: Reducing the size of data while maintaining its integrity.
- **Scientific Computing**: Solving large-scale linear systems and eigenvalue problems more efficiently.

# The project
You have to develop a C++ library that implements some of the algorithm described in the given literature. The library should use efficient linear algebra sofware as the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) or [Armadillo](https://arma.sourceforge.net/) for basic linear algebra operations.

The library should be generic with respect of the floating point type an provide a clear API. Test cases should be made to compare effectiveness of the implemented algorithms.

## Further Reading

- "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions" by Halko, Martinsson, and Tropp.
- "Randomized Numerical Linear Algebra A Perspective on the Field With an Eye to Software" by R. Murrey et al.

- "Randomized Algorithms for Matrices and Data" by Mahoney.

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
- Mahoney, M. W. (2011). Randomized algorithms for matrices and data. Foundations and Trends® in Machine Learning, 3(2), 123-224.
- R. Murray et al. (2023) Randomized Numerical Linear Algebra A Perspective on the Field With an Eye to Software, https://arxiv.org/abs/2302.11474