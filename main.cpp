#include <iostream>
#include "subroutines.h"

int main() {
    // start seeding for main program, once.
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Nearest neighbor hopping term
    double t = 1.0;
    std::cout << "nearest neighbor hopping: " << t << std::endl;

    // chemical potential term
    double mu = -0.4;
    std::cout << "chemical potential: " << mu << std::endl;

    // Local U interaction term
    double U = 0.0;
    std::cout << "local interaction U: " << U << std::endl;

    // inverse temperature
    double beta = 4.0;
    std::cout << "inverse temperature: " << beta << std::endl;

    // trotter discretization imaginary time
    double delta_tau = 0.1;
    std::cout << "trotter discretization imaginary time: " << delta_tau << std::endl; 

    // length of imaginary time
    int L_tau = beta / delta_tau;
    std::cout << "length of imaginary time: " << L_tau << std::endl;

    // length size of square lattice
    int L = 4;
    int N = L * L;
    std::cout << "total sites of square lattice: " << N << std::endl;

    // number of matrix wrap for stabilization
    // the lower the better but high cost
    int nwrap = 1;
    std::cout << "number of matrix wrap for stabilization: " << nwrap << std::endl;

    // symmetric trotter discretization or not
    bool is_symmetric = true;
    std::cout << "symmetric decomposition: " << is_symmetric << std::endl;

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
    arma::Mat<int> s = initialize_random_ising(L, L_tau);

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

    for (int l = 0; l < L_tau; l++){
        // if symmetric half-warp

        // get hs field for l

        // shuffle the sites

        // n_site_accepted set to zero

        for (int& site : shuffled_sites){
            // update ratio r_up r_dn

            // probabiility

            if (rand() < Prob){
                // add n_site accepted counter

                // flip the ising configuration

                // update green's function locally
            }
            
        }

        // acceptance rate total += n_site_accepted / N / L_tau

        // if symmetric half_warp reverse

        // recalculate Green's function for stability

        // record numerical error

        // propagate for next Green eq
        
    }

    return 0;
}


