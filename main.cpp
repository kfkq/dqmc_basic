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

    // Declare variables for input parameters
    double t, mu, U, beta, delta_tau;
    int L, nwrap, nstab, nsweep_measure, nsweep_thermal, nbins;
    bool is_symmetric, is_forward;

    // Read input parameters from file
    try {
        read_input_parameters(
            "input_dqmc.in", 
            t, mu, U, beta, delta_tau, 
            L, nwrap, nstab, nsweep_measure, nsweep_thermal, nbins, 
            is_symmetric, is_forward
        );
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Print input parameters
    std::cout << "Input parameters:" << std::endl;
    std::cout << "t = " << t << std::endl;
    std::cout << "mu = " << mu << std::endl;
    std::cout << "U = " << U << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "delta_tau = " << delta_tau << std::endl;
    std::cout << "L = " << L << std::endl;
    std::cout << "nwrap = " << nwrap << std::endl;
    std::cout << "nstab = " << nstab << std::endl;
    std::cout << "nsweep_measure = " << nsweep_measure << std::endl;
    std::cout << "nsweep_thermal = " << nsweep_thermal << std::endl;
    std::cout << "nbins = " << nbins << std::endl;
    std::cout << "is_symmetric = " << std::boolalpha << is_symmetric << std::endl;
    std::cout << "is_forward = " << std::boolalpha << is_forward << std::endl;

    // length of imaginary time
    int L_tau = beta / delta_tau;
    std::cout << "length of imaginary time: " << L_tau << std::endl;

    // total sites of square lattice
    int N = L * L;
    std::cout << "total sites of square lattice: " << N << std::endl;

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

    // do thermalization first
    std::cout << "thermalization process ..." << std::endl;
    for (int is = 0; is < nsweep_thermal; is++)
    {
        sweep_time_slices(
                Gup, Gdn, 
                expVup, expVdn, 
                expK, inv_expK, 
                s, alpha, 
                L_tau, N, nwrap, nstab, 
                is_symmetric, 
                rng, dis, 
                acceptance_rate
            );
    }

    // Open files for each measurement
    std::ofstream outfile_D("double_occupancy.txt");
    std::ofstream outfile_Ekin("kinetic_energy.txt");
    std::ofstream outfile_Epot("potential_energy.txt");

    if (!outfile_D.is_open() || !outfile_Ekin.is_open() || !outfile_Epot.is_open()) {
        std::cerr << "Error: Could not open measurement files for writing." << std::endl;
        return 1;
    }

    // Write headers to the files
    outfile_D << "Bin\tAverage\tStandard_Error\n";
    outfile_Ekin << "Bin\tAverage\tStandard_Error\n";
    outfile_Epot << "Bin\tAverage\tStandard_Error\n";

    // Measurement loop
    for (int ib = 0; ib < nbins; ib++) {
        std::vector<double> D_measurements; // Store double occupancy measurements
        std::vector<double> E_kin_measurements; // Store kinetic energy measurements
        std::vector<double> E_pot_measurements; // Store potential energy measurements

        // Perform measurements for each sweep in this bin
        for (int is = 0; is < nsweep_measure; is++) {
            sweep_time_slices(
                Gup, Gdn, 
                expVup, expVdn, 
                expK, inv_expK, 
                s, alpha, 
                L_tau, N, nwrap, nstab, 
                is_symmetric, 
                rng, dis, 
                acceptance_rate
            );

            // Collect measurements
            D_measurements.push_back(measure_double_occupancy(Gup, Gdn));
            E_kin_measurements.push_back(measure_kinetic_energy(Gup, Gdn, t, L));
            E_pot_measurements.push_back(measure_potential_energy(Gup, Gdn, U));
        }

        // Compute statistics for each measurement
        auto [D_avg, D_err] = compute_stats(D_measurements);
        auto [E_kin_avg, E_kin_err] = compute_stats(E_kin_measurements);
        auto [E_pot_avg, E_pot_err] = compute_stats(E_pot_measurements);

        // Write results to respective files
        outfile_D << ib << "\t" << D_avg << "\t" << D_err << "\n";
        outfile_Ekin << ib << "\t" << E_kin_avg << "\t" << E_kin_err << "\n";
        outfile_Epot << ib << "\t" << E_pot_avg << "\t" << E_pot_err << "\n";

        std::cout << "Bin " << ib << " completed." << std::endl;
    }

    // Close the files
    outfile_D.close();
    outfile_Ekin.close();
    outfile_Epot.close();

    acceptance_rate = acceptance_rate / (nsweep_thermal + nsweep_measure * nbins);


    return 0;
}


