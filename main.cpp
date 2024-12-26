#include <iostream>
#include "subroutines.h"

int main() {
    // start seeding for main program, once.
    std::srand(static_cast<unsigned int>(std::time(0)));

    // create a random number generator
    std::random_device rd;  // Seed generator (non-deterministic)
    std::mt19937 gen(rd()); // Mersenne Twister random number generator
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Nearest neighbor hopping term
    double t = 1.0;
    std::cout << "nearest neighbor hopping: " << t << std::endl;

    // chemical potential term
    double mu = -0.4;
    std::cout << "chemical potential: " << mu << std::endl;

    // Local U interaction term
    double U = 2.0;
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
    int nwrap = 8;
    std::cout << "number of matrix wrap for stabilization: " << nwrap << std::endl;

    // number of time slice before recalculalting G_eq from scratch
    int nstab = 5;
    std::cout << "number of time slice before recalculating G_eq from scratch: " << nstab << std::endl;

    // symmetric trotter discretization or not
    bool is_symmetric = true;
    std::cout << "symmetric decomposition: " << is_symmetric << std::endl;

    // forward propagation or not
    bool is_forward = true;
    std::cout << "forward propagation: " << is_forward << std::endl;

    if (is_symmetric)
    {
        delta_tau = 0.5 * delta_tau;
    }   

    double spin_up =  1.0;
    double spin_dn = -1.0;

    // kinetic matrix
    arma::mat K = build_Kmat(L, t, mu);

    // calculate exponential of the kinetic matrix exp(- dt * K) and exp(+ dt * K)
    // This is NxN matrices
    arma::mat expK = calculate_exp_Kmat(K, delta_tau, -1.0);
    arma::mat inv_expK = calculate_exp_Kmat(K, delta_tau, 1.0);

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

    // wrap into Bup and Bdn
    LDRMatrix Bup = wrap_B_matrices(expK, expVup, nwrap, is_symmetric);
    LDRMatrix Bdn = wrap_B_matrices(expK, expVdn, nwrap, is_symmetric);

    // calculate G_eq_00
    auto [Gup, signdetGup] = calculate_invIpA(Bup);
    auto [Gdn, signdetGdn] = calculate_invIpA(Bdn);

    double acceptance_rate = 0.0;

    for (int l = 0; l < L_tau; l++){
        // propagate forward to G_eq(\tau,\tau)
        propagate_equaltime_greens(Gup, expK, expVup, l, is_symmetric, true);
        propagate_equaltime_greens(Gdn, expK, expVdn, l, is_symmetric, true);

        // if symmetric warp
        if (is_symmetric)
        {
            symmmetric_warp_greens(Gup, expK, inv_expK, true);
            symmmetric_warp_greens(Gdn, expK, inv_expK, true);
        }

        // shuffle the sites
        std::vector<int> shuffled_sites = shuffle_numbers(s.n_rows, rng);

        int accepted = 0;
        for (int& site : shuffled_sites){
            // update ratio r_up r_dn
            double Gup_ii = Gup(site, site);
            double Gdn_ii = Gdn(site, site);
            double s_il = s(site,l);

            auto [ratio_up, delta_up] = update_ratio_hubbard(Gup_ii, s_il, alpha, spin_up);
            auto [ratio_dn, delta_dn] = update_ratio_hubbard(Gdn_ii, s_il, alpha, spin_dn);

            // probabiility
            double prob = abs(ratio_up * ratio_dn);

            if (dis(gen) < prob){
                // add n_site accepted counter
                accepted += 1;

                // flip the ising configuration
                s(site, l) = -s_il;

                // update green's function locally
                local_update_greens(Gup, expVup, ratio_up, delta_up, site, l);
                local_update_greens(Gdn, expVdn, ratio_dn, delta_dn, site, l);
            }
            
        }

        // acceptance rate total += n_site_accepted / N
        acceptance_rate = static_cast<double>(accepted) / N;

        // if symmetric warp reverse
        if (is_symmetric)
        {
            symmmetric_warp_greens(Gup, expK, inv_expK, false);
            symmmetric_warp_greens(Gdn, expK, inv_expK, false);
        }

        // recalculate Green's function for stability
        if (l % nstab == 0)
        {
            // calculate G(tau, tau) from scratch

            // shift the expVup and expVdn so the order of B(\tau,0)B(\beta,0) is correct
            arma::mat expVup_shifted = shiftMatrixColumnsLeft(expVup, l + 1);
            arma::mat expVdn_shifted = shiftMatrixColumnsLeft(expVdn, l + 1);

            // wrap into Bup and Bdn = B(\tau,0)B(\beta,0)
            LDRMatrix Bup = wrap_B_matrices(expK, expVup_shifted, nwrap, is_symmetric);
            LDRMatrix Bdn = wrap_B_matrices(expK, expVdn_shifted, nwrap, is_symmetric);

            // calculate G_eq
            auto [Gup, signdetGup] = calculate_invIpA(Bup);
            auto [Gdn, signdetGdn] = calculate_invIpA(Bdn);
        }
        
    }

    return 0;
}


