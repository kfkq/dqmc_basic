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

    // =========================
    // Test 3: wrap_B_matrices
    // =========================
    std::cout << "\n======= Test 3: wrap_B_matrices =======" << std::endl;
    try {

        int nmat = 8;
        // Create some random 5x5 matrices
        std::vector<arma::mat> B_matrices = {
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat),
            arma::randu<arma::mat>(nmat, nmat)
        };

        // Suppose we wrap them in groups of Nwrap=2
        int Nwrap = 4;
        LDRMatrix wrappedLDR = wrap_B_matrices(B_matrices, Nwrap);

        // Reconstruct the product from L, D, R
        arma::mat productLDR = wrappedLDR.L * arma::diagmat(wrappedLDR.D) * wrappedLDR.R;

        // Compute the direct product of all B_matrices
        arma::mat productDirect = arma::eye<arma::mat>(nmat, nmat);
        for (auto& B : B_matrices) {
            productDirect = B * productDirect;
        }

        // Compare
        double diff = arma::norm(productLDR - productDirect, "fro");
        std::cout << "Difference between direct product and wrapped LDR: " << diff << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed Test 3: " << e.what() << std::endl;
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
    int nwrap = 1;

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

    // initialize Bup_l and Bdn_l vectors for all time slices.
    std::vector<arma::mat> Bup_stack;
    std::vector<arma::mat> Bdn_stack;


    // Loop over time slices
    for (int l = 0; l < L_tau; ++l)
    {
        arma::mat Bup_l = calculate_B_matrix(expKmat, expVup, l, is_symmetric);
        
        // 3d. Build the B-matrix for spin down similarly
        arma::mat Bdn_l = calculate_B_matrix(expKmat, expVdn, l, is_symmetric);

        // 4. Push them into the vectors
        Bup_stack.push_back(Bup_l);
        Bdn_stack.push_back(Bdn_l);
    }

    // wrap into Bup and Bdn
    LDRMatrix Bup = wrap_B_matrices(Bup_stack, nwrap);
    LDRMatrix Bdn = wrap_B_matrices(Bdn_stack, nwrap);

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
    symmmetric_warp_greens(Gup, expKmat, inv_expKmat, true);
    std::cout << "Gup[4,4] calculated=  " << Gup(3,3) << std::endl;
    std::cout << "Gup[4,4] correct   = " << 0.6022768205143244 << std::endl;

    // test update ratio
    std::cout << "\n======= Test 6: ratio probability =======" << std::endl;
    int l = 0;
    int isite = 11;
    
    double G_ii = Gup(isite, isite);
    int s_il = s(isite, l);

    auto [r, delta] = update_ratio_hubbard(G_ii, s_il, alpha, spin_up);

    std::cout << "Ratio probability flipping calculated: " << r << std::endl;
    std::cout << "Ratio probability flipping correct   : " << 1.5897629046313408 << std::endl;

    // test update Green
    std::cout << "\n======= Test 7: update green =======" << std::endl;
    
    arma::log_det(logdetG, sign, Gup);
    std::cout << "logdetG before flip: " << logdetG << std::endl;
    std::cout << "logdetG before flip correct:" << -43.69082373753489 << std::endl;

    s(isite, l) = -s(isite,l); // flip
    local_update_greens(Gup, expVup, r, delta, isite, l);

    arma::log_det(logdetG, sign, Gup);
    std::cout << "logdetG before flip: " << logdetG << std::endl;
    std::cout << "logdetG after flip correct: " << -44.15440862606399 << std::endl;

    // test propagate
    std::cout << "\n======= Test 6: forward propagation equal time Green's function =======" << std::endl;
    propagate_equaltime_greens(Gup, Bup_stack, expKmat, expVup, 1);
    arma::log_det(logdetG, sign, Gup);
    std::cout << "logdetG calculated = " << logdetG << std::endl;
    std::cout << "logdetG correct    = " << -43.69082373753489 << std::endl;

    

    return 0;
}