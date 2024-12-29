#include <iostream>
#include <chrono> 
#include <armadillo>
#include <tuple>
#include "subroutines.h"

// Function to test build_Kmat
void test_build_Kmat() {
    std::cout << "======= Testing build_Kmat =======" << std::endl;

    // Test parameters
    int L = 2;          // Lattice size (2x2)
    double t = 1.0;     // Hopping parameter
    double mu = -0.4;   // Chemical potential

    // Expected result (manually computed for L=2)
    arma::mat expected_K = {
        {-mu, -t, -t,  0},
        {-t, -mu,  0, -t},
        {-t,  0, -mu, -t},
        { 0, -t, -t, -mu}
    };

    // Call the function to test
    arma::mat K = build_Kmat(L, t, mu);

    // Check if the result matches the expected matrix
    if (arma::approx_equal(K, expected_K, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: build_Kmat produced the correct matrix." << std::endl;
    } else {
        std::cout << "Test FAILED: build_Kmat did not produce the expected matrix." << std::endl;
        std::cout << "Expected matrix:" << std::endl << expected_K << std::endl;
        std::cout << "Computed matrix:" << std::endl << K << std::endl;
    }
}

// Function to test calculate_exp_Kmat
void test_calculate_exp_Kmat() {
    std::cout << "\n======= Testing calculate_exp_Kmat =======" << std::endl;

    // Test parameters
    int L = 2;          // Lattice size (2x2)
    double t = 1.0;     // Hopping parameter
    double mu = -0.4;   // Chemical potential
    double delta_tau = 0.1; // Time step
    double sign = -1.0; // Sign for the exponential

    // Build the kinetic matrix
    arma::mat K = build_Kmat(L, t, mu);

    // Call the function to test
    arma::mat expK = calculate_exp_Kmat(K, delta_tau, sign);

    // Expected result (computed using Armadillo's expmat for verification)
    arma::mat expected_expK = arma::expmat(sign * delta_tau * K);

    // Check if the result matches the expected matrix
    if (arma::approx_equal(expK, expected_expK, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: calculate_exp_Kmat produced the correct matrix." << std::endl;
    } else {
        std::cout << "Test FAILED: calculate_exp_Kmat did not produce the expected matrix." << std::endl;
        std::cout << "Expected matrix:" << std::endl << expected_expK << std::endl;
        std::cout << "Computed matrix:" << std::endl << expK << std::endl;
    }
}

// Function to test initialize_random_ising
void test_initialize_random_ising() {
    std::cout << "\n======= Testing initialize_random_ising =======" << std::endl;

    // Test parameters
    int L = 2;          // Lattice size (2x2)
    int L_tau = 10;     // Number of time slices

    // Call the function to test
    arma::Mat<int> s = initialize_random_ising(L, L_tau);

    // Check the dimensions of the matrix
    if (s.n_rows == L * L && s.n_cols == L_tau) {
        std::cout << "Test PASSED: Matrix dimensions are correct." << std::endl;
    } else {
        std::cout << "Test FAILED: Matrix dimensions are incorrect." << std::endl;
        std::cout << "Expected dimensions: (" << L * L << ", " << L_tau << ")" << std::endl;
        std::cout << "Computed dimensions: (" << s.n_rows << ", " << s.n_cols << ")" << std::endl;
    }

    // Check if all elements are either +1 or -1
    bool valid_spins = true;
    for (int i = 0; i < s.n_rows; ++i) {
        for (int j = 0; j < s.n_cols; ++j) {
            if (s(i, j) != 1 && s(i, j) != -1) {
                valid_spins = false;
                break;
            }
        }
        if (!valid_spins) break;
    }

    if (valid_spins) {
        std::cout << "Test PASSED: All spins are either +1 or -1." << std::endl;
    } else {
        std::cout << "Test FAILED: Invalid spin values detected." << std::endl;
    }
}

void test_calculate_exp_Vmat() {
    std::cout << "\n======= Testing calculate_exp_Vmat =======" << std::endl;

    // Test parameters
    double sgn = 1.0;           // Sign for spin-up or spin-down
    double alpha = 0.5;         // Hubbard-Stratonovich parameter
    int L = 2;                  // Lattice size (2x2)
    int L_tau = 10;             // Number of time slices

    // Create a fixed Ising configuration for testing
    arma::Mat<int> s = {
        {1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
        {-1, 1, -1, 1, -1, 1, -1, 1, -1, 1},
        {1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
        {-1, 1, -1, 1, -1, 1, -1, 1, -1, 1}
    };

    // Call the function to test
    arma::mat expV = calculate_exp_Vmat(sgn, alpha, s);

    // Expected result (manually computed)
    arma::mat expected_expV = {
        {std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0)},
        {std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0)},
        {std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0)},
        {std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0), std::exp(alpha * -1.0), std::exp(alpha * 1.0)}
    };

    // Check if the result matches the expected matrix
    if (arma::approx_equal(expV, expected_expV, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: calculate_exp_Vmat produced the correct matrix." << std::endl;
    } else {
        std::cout << "Test FAILED: calculate_exp_Vmat did not produce the expected matrix." << std::endl;
        std::cout << "Expected matrix:" << std::endl << expected_expV << std::endl;
        std::cout << "Computed matrix:" << std::endl << expV << std::endl;
    }
}

#include <chrono> // For timing

void test_wrap_B_matrices() {
    std::cout << "\n======= Testing wrap_B_matrices =======" << std::endl;

    // Test parameters
    int L = 20;                  // Lattice size (8x8, larger for testing)
    double t = 1.0;             // Hopping parameter
    double mu = -0.4;           // Chemical potential
    double delta_tau = 0.1;     // Time step
    int nwrap = 10;             // Number of matrices to wrap
    bool is_symmetric = true;   // Symmetric Trotter decomposition

    // Build the kinetic matrix and its exponential
    arma::mat K = build_Kmat(L, t, mu);
    arma::mat expK = calculate_exp_Kmat(K, delta_tau, -1.0);

    // Initialize expV as a random matrix
    arma::mat expV = arma::randu<arma::mat>(L * L, nwrap);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to test
    LDRMatrix Bwrap = wrap_B_matrices(expK, expV, nwrap, is_symmetric);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to compute wrap_B_matrices for L = 20: " << elapsed.count() << " seconds" << std::endl;

    // Check the dimensions of the LDR components
    bool valid_dimensions = (Bwrap.L.n_rows == L * L && Bwrap.L.n_cols == L * L) &&
                            (Bwrap.D.n_elem == L * L) &&
                            (Bwrap.R.n_rows == L * L && Bwrap.R.n_cols == L * L);

    if (valid_dimensions) {
        std::cout << "Test PASSED: LDR components have correct dimensions." << std::endl;
    } else {
        std::cout << "Test FAILED: LDR components have incorrect dimensions." << std::endl;
    }

    // Reconstruct the matrix from LDR components
    arma::mat reconstructed = Bwrap.L * arma::diagmat(Bwrap.D) * Bwrap.R;

    // Compute the product of the B matrices manually for comparison
    arma::mat B_product = arma::eye<arma::mat>(L * L, L * L);
    for (int i = 0; i < nwrap; ++i) {
        arma::mat B = calculate_B_matrix(expK, expV, i, is_symmetric);
        B_product = B * B_product;
    }

    // Check if the reconstructed matrix matches the product of B matrices
    if (arma::approx_equal(reconstructed, B_product, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: LDR decomposition correctly reconstructs the matrix." << std::endl;
    } else {
        std::cout << "Test FAILED: LDR decomposition does not match the product of B matrices." << std::endl;
        std::cout << "Reconstructed matrix:" << std::endl << reconstructed << std::endl;
        std::cout << "Expected matrix:" << std::endl << B_product << std::endl;
    }
}

void test_calculate_invIpA() {
    std::cout << "\n======= Testing calculate_invIpA =======" << std::endl;

    // Test parameters
    int size = 20; // Size of the matrix

    // Create a random matrix A
    arma::mat A = arma::randu<arma::mat>(size, size);

    // Perform LDR decomposition of A
    LDRMatrix A_LDR = qr_LDR(A);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to test
    auto [invResult, sgnDetResult] = calculate_invIpA(A_LDR);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to compute calculate_invIpA for L = 20: " << elapsed.count() << " seconds" << std::endl;

    // Expected result: Direct inverse of (I + A)
    arma::mat I = arma::eye<arma::mat>(size, size);
    arma::mat expected_inv = arma::inv(I + A);

    // Check if the result matches the expected inverse
    if (arma::approx_equal(invResult, expected_inv, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: calculate_invIpA produced the correct inverse." << std::endl;
    } else {
        std::cout << "Test FAILED: calculate_invIpA did not produce the expected inverse." << std::endl;
        std::cout << "Expected inverse:" << std::endl << expected_inv << std::endl;
        std::cout << "Computed inverse:" << std::endl << invResult << std::endl;
    }

    // Check the sign of the determinant
    double expected_sgnDet = arma::sign(arma::det(expected_inv));
    if (std::abs(sgnDetResult - expected_sgnDet) < 1e-6) {
        std::cout << "Test PASSED: Sign of the determinant is correct." << std::endl;
    } else {
        std::cout << "Test FAILED: Sign of the determinant is incorrect." << std::endl;
        std::cout << "Expected sign: " << expected_sgnDet << std::endl;
        std::cout << "Computed sign: " << sgnDetResult << std::endl;
    }
}

void test_propagate_equaltime_greens() {
    std::cout << "\n======= Testing propagate_equaltime_greens =======" << std::endl;

    // Test parameters for larger matrices
    int L = 20;                  // Lattice size (8x8, larger for testing)
    double t = 1.0;             // Hopping parameter
    double mu = -0.4;           // Chemical potential
    double delta_tau = 0.1;     // Time step
    bool is_symmetric = true;   // Symmetric Trotter decomposition
    bool forward = true;        // Forward propagation

    // Build the kinetic matrix and its exponential
    arma::mat K = build_Kmat(L, t, mu);
    arma::mat expK = calculate_exp_Kmat(K, delta_tau, -1.0);

    // Initialize expV as a random matrix
    arma::mat expV = arma::randu<arma::mat>(L * L, 10); // 10 time slices

    // Initialize the Green's function as a random matrix
    arma::mat G = arma::randu<arma::mat>(L * L, L * L);

    // Expected result: G = B * G * B^{-1}, where B = expK * diag(expV) * expK
    arma::mat B = calculate_B_matrix(expK, expV, 0, is_symmetric);
    arma::mat inv_B = arma::inv(B);
    arma::mat expected_G = B * G * inv_B;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to test
    propagate_equaltime_greens(G, expK, expV, 0, is_symmetric, forward);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to compute propagate_equaltime_greens for L = 20: " << elapsed.count() << " seconds" << std::endl;

    // Check if the result matches the expected Green's function
    if (arma::approx_equal(G, expected_G, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: propagate_equaltime_greens produced the correct Green's function." << std::endl;
    } else {
        std::cout << "Test FAILED: propagate_equaltime_greens did not produce the expected Green's function." << std::endl;
        std::cout << "Expected Green's function:" << std::endl << expected_G << std::endl;
        std::cout << "Computed Green's function:" << std::endl << G << std::endl;
    }
}

void test_symmmetric_warp_greens() {
    std::cout << "\n======= Testing symmmetric_warp_greens =======" << std::endl;

    // Test parameters for larger matrices
    int L = 20;                  // Lattice size (8x8, larger for testing)
    double t = 1.0;             // Hopping parameter
    double mu = -0.4;           // Chemical potential
    double delta_tau = 0.1;     // Time step
    bool forward = true;        // Forward propagation

    // Build the kinetic matrix and its exponential
    arma::mat K = build_Kmat(L, t, mu);
    arma::mat expK = calculate_exp_Kmat(K, delta_tau, -1.0);
    arma::mat inv_expK = calculate_exp_Kmat(K, delta_tau, 1.0);

    // Initialize the Green's function as a random matrix
    arma::mat G = arma::randu<arma::mat>(L * L, L * L);

    // Expected result: G = inv(expK) * G * expK (for forward propagation)
    arma::mat expected_G = inv_expK * G * expK;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to test
    symmmetric_warp_greens(G, expK, inv_expK, forward);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to compute symmmetric_warp_greens for L = 20: " << elapsed.count() << " seconds" << std::endl;

    // Check if the result matches the expected Green's function
    if (arma::approx_equal(G, expected_G, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: symmmetric_warp_greens produced the correct Green's function." << std::endl;
    } else {
        std::cout << "Test FAILED: symmmetric_warp_greens did not produce the expected Green's function." << std::endl;
        std::cout << "Expected Green's function:" << std::endl << expected_G << std::endl;
        std::cout << "Computed Green's function:" << std::endl << G << std::endl;
    }
}

#include <chrono> // For timing

void test_local_update_greens() {
    std::cout << "\n======= Testing local_update_greens =======" << std::endl;

    // Test parameters for larger matrices
    int L = 20;                  // Lattice size (8x8, larger for testing)
    int N = L * L;              // Total number of sites
    int l = 0;                  // Time slice index
    double delta = 0.1;         // Change in on-site energy
    double r = 1.5;             // Determinant ratio
    int i = 0;                  // Site index to update

    // Initialize the Green's function as a random matrix
    arma::mat G = arma::randu<arma::mat>(N, N);

    // Initialize the potential matrix as a random matrix
    arma::mat expV = arma::randu<arma::mat>(N, 10); // 10 time slices

    // Expected result: G = G + (delta / r) * u * v.t()
    arma::vec u = G.col(i);
    arma::vec v = G.row(i).t();
    v(i) = v(i) - 1.0; // Subtract the identity term for v(i)
    arma::mat expected_G = G + (delta / r) * (u * v.t());

    // Expected result: expV(i, l) = expV(i, l) * (1.0 + delta)
    double expected_expV = expV(i, l) * (1.0 + delta);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to test
    local_update_greens(G, expV, r, delta, i, l);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to compute local_update_greens for L = 20: " << elapsed.count() << " seconds" << std::endl;

    // Check if the Green's function was updated correctly
    if (arma::approx_equal(G, expected_G, "absdiff", 1e-6)) {
        std::cout << "Test PASSED: Green's function was updated correctly." << std::endl;
    } else {
        std::cout << "Test FAILED: Green's function was not updated correctly." << std::endl;
        std::cout << "Expected Green's function:" << std::endl << expected_G << std::endl;
        std::cout << "Computed Green's function:" << std::endl << G << std::endl;
    }

    // Check if the potential matrix was updated correctly
    if (std::abs(expV(i, l) - expected_expV) < 1e-6) {
        std::cout << "Test PASSED: Potential matrix was updated correctly." << std::endl;
    } else {
        std::cout << "Test FAILED: Potential matrix was not updated correctly." << std::endl;
        std::cout << "Expected expV(i, l): " << expected_expV << std::endl;
        std::cout << "Computed expV(i, l): " << expV(i, l) << std::endl;
    }
}

#include <chrono> // For timing

void test_update_ratio_hubbard() {
    std::cout << "\n======= Testing update_ratio_hubbard =======" << std::endl;

    // Test parameters
    double G_ii = 0.5;          // Diagonal element of the Green's function
    int s_il = 1;               // Ising spin at site i and time slice l
    double alpha = 0.5;         // Hubbard-Stratonovich parameter
    double sgn = 1.0;           // Sign for spin-up or spin-down

    // Expected result: ratio = 1 + delta * (1 - G_ii), delta = exp(sgn * -2.0 * alpha * s_il) - 1.0
    double expected_delta = std::exp(sgn * -2.0 * alpha * static_cast<double>(s_il)) - 1.0;
    double expected_ratio = 1.0 + expected_delta * (1.0 - G_ii);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to test
    auto [ratio, delta] = update_ratio_hubbard(G_ii, s_il, alpha, sgn);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to compute update_ratio_hubbard: " << elapsed.count() << " seconds" << std::endl;

    // Check if the result matches the expected ratio and delta
    if (std::abs(ratio - expected_ratio) < 1e-6 && std::abs(delta - expected_delta) < 1e-6) {
        std::cout << "Test PASSED: update_ratio_hubbard produced the correct ratio and delta." << std::endl;
    } else {
        std::cout << "Test FAILED: update_ratio_hubbard did not produce the expected ratio and delta." << std::endl;
        std::cout << "Expected ratio: " << expected_ratio << ", Computed ratio: " << ratio << std::endl;
        std::cout << "Expected delta: " << expected_delta << ", Computed delta: " << delta << std::endl;
    }
}


// Function to diagonalize the kinetic matrix H0 = K
std::tuple<arma::vec, arma::mat> diagonalize_H0(const arma::mat& K) {
    // Step 1: Diagonalize K to obtain eigenvalues and eigenvectors
    arma::vec epsilon; // Eigenvalues
    arma::mat U;       // Eigenvectors
    arma::eig_sym(epsilon, U, K);

    // Step 2: Return eigenvalues and eigenvectors
    return std::make_tuple(epsilon, U);
}

// Function to compute the retarded Green's function exactly in non-interacting
arma::mat retarded_greens(double tau, double beta, const arma::vec& epsilon, const arma::mat& U) {
    // Step 1: Compute gτ = 1 / (exp(τ * ϵ) + exp((τ - β) * ϵ))
    arma::vec g_tau = 1.0 / (arma::exp(tau * epsilon) + arma::exp((tau - beta) * epsilon));

    // Step 2: Compute Gτ = U * Diagonal(gτ) * U^†
    arma::mat G_tau = U * arma::diagmat(g_tau) * U.t();

    // Return Gτ
    return G_tau;
}

void test_dqmc_vs_exact() {
    std::cout << "\n======= Testing DQMC vs Exact Green's Function =======" << std::endl;

    // Step 1: Initialize parameters
    int L = 4;                  // Lattice size (4x4)
    double t = 1.0;             // Hopping parameter
    double mu = -0.4;           // Chemical potential
    double beta = 4.0;          // Inverse temperature
    double delta_tau = 0.1;     // Imaginary time step
    int nwrap = 10;              // Number of matrix wraps
    bool is_symmetric = false;   // Symmetric Trotter decomposition

    // Step 2: Build the kinetic matrix K
    arma::mat K = build_Kmat(L, t, mu);

    // Step 3: Diagonalize K to get eigenvalues (epsilon) and eigenvectors (U)
    auto [epsilon, U] = diagonalize_H0(K);

    // Step 4: Initialize expK and expV
    arma::mat expK = calculate_exp_Kmat(K, delta_tau, -1.0); // exp(-Δτ K)
    arma::mat expV = arma::ones<arma::mat>(L * L, beta / delta_tau); // exp(V) = 1 for U = 0

    // Step 5: Compute G using DQMC routine
    // Wrap B matrices
    LDRMatrix Bup = wrap_B_matrices(expK, expV, nwrap, is_symmetric);

    // Compute G using invIpA
    auto [Gup, signdetGup] = calculate_invIpA(Bup);

    // For simplicity, we'll compare Gup (spin-up Green's function)
    arma::mat G_dqmc = Gup;

    // Step 6: Compute G_exact using retarded_greens
    double tau = 0.0; // Equal-time Green's function (τ = 0)
    arma::mat G_exact = retarded_greens(tau, beta, epsilon, U);

    // Step 7: Compare G_dqmc and G_exact
    double error = arma::norm(G_dqmc - G_exact, "fro"); // Frobenius norm of the difference
    double tolerance = 1e-6;

    std::cout << "Frobenius norm of the difference: " << error << std::endl;
    if (error < tolerance) {
        std::cout << "Test PASSED: DQMC Green's function matches the exact result." << std::endl;
    } else {
        std::cout << "Test FAILED: DQMC Green's function does not match the exact result." << std::endl;
        std::cout << "DQMC Green's function:" << std::endl << G_dqmc << std::endl;
        std::cout << "Exact Green's function:" << std::endl << G_exact << std::endl;
    }
}

// Main function to run all tests
int main() {
    test_build_Kmat();

    test_calculate_exp_Kmat();

    test_initialize_random_ising();

    test_calculate_exp_Vmat();

    test_wrap_B_matrices();

    test_calculate_invIpA();

    test_propagate_equaltime_greens();

    test_symmmetric_warp_greens();

    test_local_update_greens();

    test_update_ratio_hubbard();

    test_dqmc_vs_exact();

    return 0;
}