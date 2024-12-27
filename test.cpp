#include <iostream>
#include "subroutines.h"

// ------------------------------------------------------
// DQMC TEST FUNCTION
// ------------------------------------------------------

arma::Mat<int> create_fixed_conf() {
    arma::Mat<int> matrix(16, 40, arma::fill::none);

    // Fill the matrix with a specific pattern of -1 and 1
    for (arma::uword i = 0; i < matrix.n_rows; ++i) {
        for (arma::uword j = 0; j < matrix.n_cols; ++j) {
            // Example pattern: alternate -1 and 1
            matrix(i, j) = ((i + j) % 2 == 0) ? 1 : -1;
        }
    }

    return matrix;
}


// ------------------------------------------------------
// Test Program
// ------------------------------------------------------

int main() {
    // =========================
    // Test 1: qr_LDR
    // =========================
    std::cout << "======= Test 1: qr_LDR =======" << std::endl;
    try {
        // Create a random matrix A
        arma::mat A = arma::randu<arma::mat>(100, 100);

        // Perform LDR decomposition
        LDRMatrix A_LDR = qr_LDR(A);

        // Reconstruct A from L, D, R
        arma::mat A_reconstructed = A_LDR.L * arma::diagmat(A_LDR.D) * A_LDR.R;

        // Print difference (Frobenius norm of A - LDR)
        double diff = arma::norm(A - A_reconstructed, "fro");
        std::cout << "Difference between A and L*D*R: " << diff << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed Test 1: " << e.what() << std::endl;
    }

    // =========================
    // Test 2: calculate_invIpA
    // =========================
    std::cout << "\n======= Test 2: calculate_invIpA =======" << std::endl;
    try {
        // Create a random matrix A
        arma::mat A = arma::randu<arma::mat>(5, 5);

        // Compute LDR
        LDRMatrix A_LDR = qr_LDR(A);

        // Calculate our (I + A)^{-1} approximation
        auto [invResult, sgnDetResult] = calculate_invIpA(A_LDR);

        // Compare with direct inverse of (I + A)
        arma::mat directInv = arma::inv(arma::eye<arma::mat>(A.n_rows, A.n_cols) + A);

        // Compute and print Frobenius norm of the difference
        double diff = arma::norm(invResult - directInv, "fro");
        std::cout << "Difference between direct inverse and calculate_invIpA: " << diff << std::endl;
        std::cout << "Sign of determinant of the calculate_invIpA result: " << sgnDetResult << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed Test 2: " << e.what() << std::endl;
    }

    //===============================
    // DQMC ROUTINE TEST
    //==============================

    // Nearest neighbor hopping term
    double t = 1.0;

    // chemical potential term
    double mu = -0.4;

    // Local U interaction term
    double U = 2.0;

    // inverse temperature
    double beta = 4.0;

    // trotter discretization imaginary time
    double delta_tau = 0.1;

    // length of imaginary time
    int L_tau = beta / delta_tau;

    // length size of square lattice
    int L = 4;
    int N = L * L;

    // number of matrix wrap for stabilization
    // the lower the better but high cost
    int nwrap = 10;

    // symmetric trotter discretization or not
    bool is_symmetric = true;

    if (is_symmetric)
    {
        delta_tau = 0.5 * delta_tau;
    }   

    double spin_up =  1.0;
    double spin_dn = -1.0;

    // kinetic matrix
    arma::mat Kmat = build_Kmat(L, t, mu);

    // calculate exponential of the kinetic matrix exp(- dt * K) and exp(+ dt * K)
    // This is NxN matrices
    arma::mat expKmat = calculate_exp_Kmat(Kmat, delta_tau, -1.0);
    arma::mat inv_expKmat = calculate_exp_Kmat(Kmat, delta_tau, 1.0);

    // hubbard stratonovich for hubbard U
    // define constant alpha associated from HS transformation
    double alpha = 0.0;
    if (is_symmetric) {
        alpha = acosh(exp(delta_tau * U));
    } else {
        alpha = acosh(exp(0.5 * delta_tau * U));
    }      

    // initialize random ising configuration
    //arma::Mat<int> s = initialize_random_ising(L, L_tau);
    arma::Mat<int> s = create_fixed_conf();

    // calculate the expVup and expVdn
    arma::mat expVup = calculate_exp_Vmat(spin_up, alpha, s);
    arma::mat expVdn = calculate_exp_Vmat(spin_dn, alpha, s);

    // wrap into Bup and Bdn
    LDRMatrix Bup = wrap_B_matrices(expKmat, expVup, nwrap, is_symmetric);
    LDRMatrix Bdn = wrap_B_matrices(expKmat, expVdn, nwrap, is_symmetric);

    // calculate G_eq
    auto [Gup, signdetGup] = calculate_invIpA(Bup);
    auto [Gdn, signdetGdn] = calculate_invIpA(Bdn);

    // print result
    std::cout << "\n======= Test 4: Equal time Green's function =======" << std::endl;
    double logdetG = 0.0;
    double sign    = 0.0;
    arma::log_det(logdetG, sign, Gup);
    std::cout << "logdetG calculated =  " << logdetG << std::endl;
    std::cout << "logdetG correct    = " << -43.69082373753489 << std::endl;

     // test warp
    std::cout << "\n======= Test 5: symmetric warp green function =======" << std::endl;
    propagate_equaltime_greens(Gup, expKmat, expVup, 0, is_symmetric, true);
    propagate_equaltime_greens(Gdn, expKmat, expVdn, 0, is_symmetric, true);

    symmmetric_warp_greens(Gup, expKmat, inv_expKmat, true);
    symmmetric_warp_greens(Gdn, expKmat, inv_expKmat, true);
    std::cout << "logabsdetG after warp =  " << arma::log_det(Gup) << std::endl;

    // test update ratio
    std::cout << "\n======= Test 6: ratio probability =======" << std::endl;
    int l = 0;
    int isite = 0;
    
    double Gup_ii = Gup(isite, isite);
    double Gdn_ii = Gdn(isite, isite);
    int s_il = s(isite, l);

    std::cout << "sil: " << s_il << std::endl;

    auto [rup, delta_up] = update_ratio_hubbard(Gup_ii, s_il, alpha, spin_up);
    auto [rdn, delta_dn] = update_ratio_hubbard(Gdn_ii, s_il, alpha, spin_dn);

    std::cout << "Ratio probability flipping calculated: " << rup*rdn << std::endl;

    // test update ratio
    std::cout << "\n======= Test 7: update green =======" << std::endl;
    local_update_greens(Gup, expVup, rup, delta_up, isite, l);
    local_update_greens(Gdn, expVdn, rdn, delta_dn, isite, l);
    std::cout << "logabsdetG after warp =  " << arma::log_det(Gup) << std::endl;

    std::cout << "\n======= Test 8: reverse warp =======" << std::endl;
    symmmetric_warp_greens(Gup, expKmat, inv_expKmat, false);
    symmmetric_warp_greens(Gdn, expKmat, inv_expKmat, false);
    std::cout << "logabsdetG after warp =  " << arma::log_det(Gup) << std::endl;

     // Create a sample matrix
    std::cout << "\n======= Test 9: matrix shift =======" << std::endl;
    arma::mat matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    std::cout << "Original Matrix:\n" << matrix << std::endl;

    // Shift columns to the left by 1 position
    int k = 1;
    arma::mat shiftedMatrix = shiftMatrixColumnsLeft(matrix, k);

    std::cout << "Matrix after shifting columns left by " << k << ":\n" << shiftedMatrix << std::endl;

    // // test update Green
    // std::cout << "\n======= Test 7: update green =======" << std::endl;
    
    // arma::log_det(logdetG, sign, Gup);
    // std::cout << "logdetG before flip: " << logdetG << std::endl;
    // std::cout << "logdetG before flip correct:" << -43.69082373753489 << std::endl;

    // s(isite, l) = -s(isite,l); // flip
    // local_update_greens(Gup, expVup, r, delta, isite, l);

    // arma::log_det(logdetG, sign, Gup);
    // std::cout << "logdetG before flip: " << logdetG << std::endl;
    // std::cout << "logdetG after flip correct: " << -44.15440862606399 << std::endl;

    // // test propagate
    // std::cout << "\n======= Test 6: forward propagation equal time Green's function =======" << std::endl;

    

    return 0;
}