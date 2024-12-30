#include "subroutines.h"

// ------------------------------------------------------
// LINEAR ALGEBRA SUBROUTINE
// ------------------------------------------------------

template <typename T>
int sign(T value) {
    // Check if the value is positive
    // If the value is greater than zero, return 1.
    // If the value is zero, this evaluates to 0.
    // If the value is less than zero, subtracts 1 (i.e., returns -1).
    return (value > 0) - (value < 0);
}


std::vector<int> shuffle_numbers(int N, std::mt19937 rng) {
    // Step 1: Create a vector containing numbers from 1 to N
    std::vector<int> numbers(N);
    for (int i = 0; i < N; ++i) {
        numbers[i] = i; // Fill with 1, 2, ..., N
    }

    // Step 2: Shuffle the vector using std::shuffle
    std::shuffle(numbers.begin(), numbers.end(), rng);

    return numbers;
}

arma::mat shiftMatrixColumnsLeft(const arma::mat& matrix, int k) {
    if (matrix.is_empty()) return arma::mat();
    
    int cols = matrix.n_cols;
    
    // Normalize k to be within matrix size
    k = k % cols;
    if (k == 0) return matrix;
    
    // Create new matrix for the result
    arma::mat result = matrix;
    
    // Perform the shift
    for (size_t j = 0; j < cols; j++) {
        // Calculate new column position after left shift
        size_t newCol = (cols + j - k) % cols;
        result.col(newCol) = matrix.col(j);
    }
    
    return result;
}

LDRMatrix qr_LDR(arma::mat& M) {
    // Matrices for storing the QR decomposition
    arma::mat Q, R_temp;   // Q: Orthogonal matrix, R_temp: Triangular matrix from QR
    arma::uvec P_vec;      // Permutation vector from pivoted QR decomposition
    arma::vec D;           // Diagonal elements of R (absolute values)

    // Step 1: Perform pivoted QR decomposition of A.
    //         The "vector" option stores the permutation as a vector (P_vec).
    //         If the decomposition fails, throw an exception.
    bool qr_success = arma::qr(Q, R_temp, P_vec, M, "vector");
    if (!qr_success) {
        // If the QR decomposition fails, throw a runtime error with a descriptive message.
        throw std::runtime_error("QR decomposition of matrix M failed. Ensure M is well-conditioned.");
    }

    // Step 2: Extract diagonal elements of R_temp as D (absolute values).
    //         This represents the scaling factor for normalizing R_temp.
    D = arma::abs(R_temp.diag());

    // Step 3: Normalize R_temp to get R.
    //         Create a diagonal matrix with 1 / D and multiply it with R_temp.
    arma::mat D_inv = arma::diagmat(1 / D);  // Diagonal inverse of D
    arma::mat R = D_inv * R_temp;           // Normalize R_temp to get R

    // Step 4: Adjust the columns of R to their original order using P_vec.
    //         Pivoting may have altered the column order; this step restores it.
    arma::uvec P_inv = arma::sort_index(P_vec);  // Compute the inverse permutation
    R = R.cols(P_inv);                          // Permute R back to the original order

    // Step 5: Return the LDRMatrix struct containing Q, D, and R.
    return LDRMatrix{Q, D, R};
}

std::pair<arma::mat, double> calculate_invIpA(LDRMatrix& A_LDR) {
    // Step 1: Split the diagonal vector D into D_min and D_max.
    // D_min contains element-wise min(D, 1).
    // D_max contains element-wise max(D, 1).
    arma::vec D_min = arma::min(A_LDR.D, arma::vec(A_LDR.D.n_elem, arma::fill::ones)); // min(D, 1)
    arma::vec D_max = arma::max(A_LDR.D, arma::vec(A_LDR.D.n_elem, arma::fill::ones)); // max(D, 1)

    // Step 2: Compute the inverse of D_max as a diagonal matrix.
    arma::mat D_max_inv = arma::diagmat(1 / D_max);

    // Step 3: Compute \( \mathbf{R}^{-1} \cdot \mathbf{D}_{\text{max}}^{-1} \).
    // This step combines the inverse of \( \mathbf{R} \) with the inverse of \( \mathbf{D}_{\text{max}} \).
    arma::mat R_inv = arma::inv(A_LDR.R);       // Inverse of the upper triangular matrix R
    arma::mat Ra_Dmax_inv = R_inv * D_max_inv;  // Combine R_inv with D_max_inv

    // Step 4: Compute the intermediate matrix \( \mathbf{M} = \mathbf{R}_a^{-1} \cdot \mathbf{D}_{\text{max}}^{-1} + \mathbf{L} \cdot \mathbf{D}_{\text{min}} \).
    // \( \mathbf{M} \) is used to stabilize the computation of \( (\mathbf{I} + \mathbf{A})^{-1} \).
    arma::mat M = Ra_Dmax_inv + A_LDR.L * arma::diagmat(D_min);

    // Step 5: Compute the inverse of \( \mathbf{M} \).
    arma::mat M_inv = arma::inv(M);

    // Step 6: Compute the result \( (\mathbf{I} + \mathbf{A})^{-1} \) as \( \mathbf{R}_a^{-1} \cdot \mathbf{D}_{\text{max}}^{-1} \cdot \mathbf{M}^{-1} \).
    arma::mat result = Ra_Dmax_inv * M_inv;

    // Step 7: Compute the sign of the determinant of the resulting matrix.
    // The determinant is used for numerical stability checks.
    double sgnDetResult = sign(arma::det(result));

    // Step 8: Return the resulting matrix and the sign of its determinant.
    return std::make_pair(result, sgnDetResult);
}

arma::mat calculate_B_matrix(const arma::mat& expK, arma::mat expV, int time_slice, bool is_symmetric) {
    // Step 1: Extract the diagonal vector from expV for the given time slice.
    //         - `expV` is a matrix where each column corresponds to a time slice.
    //         - `diag_expV` is the diagonal vector for the current time slice.
    arma::vec diag_expV = expV.col(time_slice);

    // Step 2: Construct a diagonal matrix from the diagonal vector diag_expV.
    //         - This represents the potential energy contribution for the current time slice.
    arma::mat diagExpV = arma::diagmat(diag_expV);

    arma::mat B; // Initialize the B-matrix.

    // Step 3: Compute the B-matrix based on the Trotter decomposition type.
    if (is_symmetric) {
        // Symmetric Trotter decomposition:
        //   B = exp(-dt/2 * K) * diag(expV) * exp(-dt/2 * K)
        //   - `expmKmat`: Represents exp(-dt/2 * K) (precomputed outside this function).
        //   - `diagExpV`: Diagonal matrix for the potential term.
        B = expK * diagExpV * expK;
    } else {
        // Asymmetric Trotter decomposition:
        //   B = exp(-dt * K) * diag(expV)
        //   - `expmKmat`: Represents exp(-dt * K).
        //   - `diagExpV`: Diagonal matrix for the potential term.
        B = expK * diagExpV;
    }

    // Step 4: Return the computed B-matrix.
    return B;
}

LDRMatrix wrap_B_matrices(const arma::mat& expK, arma::mat& expV, int Nwrap, bool is_symmetric) {
    // Step 0: Validate input.
    int L = expV.n_cols; // Total number of matrices
    if (L % Nwrap != 0) {
        // Ensure the total number of matrices L is divisible by Nwrap.
        throw std::invalid_argument("L (number of matrices) must be divisible by Nwrap.");
    }

    int nsize = expV.n_rows; // Size of the square matrices (assumes consistent dimensions)
    int num_groups = L / Nwrap;      // Number of groups formed by Nwrap

    arma::mat B_l;

    // Step 1: Compute the product of each group.
    //         Multiply the matrices in groups of Nwrap and store the results in Bgroup.
    std::vector<arma::mat> Bgroup(num_groups);
    for (int i = 0; i < num_groups; ++i) {
        arma::mat B_dum = arma::eye<arma::mat>(nsize, nsize); // Start with an identity matrix
        for (int j = 0; j < Nwrap; ++j) {
            // Multiply matrices in the current group: B_dum = B_{i * Nwrap} * ... * B_{i * Nwrap + Nwrap-1}
            B_l = calculate_B_matrix(expK, expV, i*Nwrap+j, is_symmetric);
            B_dum = B_l * B_dum;
        }
        Bgroup[i] = B_dum; // Store the product for this group
    }

    // Step 2: Initialize an LDRMatrix to represent the "wrapped" result.
    //         Start with identity matrices for L, R and ones for D.
    LDRMatrix Bwrap;
    Bwrap.L = arma::eye<arma::mat>(nsize, nsize); // Lower triangular matrix initialized to identity
    Bwrap.D = arma::ones<arma::vec>(nsize);       // Diagonal elements initialized to ones
    Bwrap.R = arma::eye<arma::mat>(nsize, nsize); // Upper triangular matrix initialized to identity

    // Step 3: Iteratively process each group and compute the wrapped LDR representation.
    for (int i = 0; i < num_groups; ++i) {
        // Perform LDR multiplication:
        // Compute M = Bgroup[i] * (Lwrap * diag(Dwrap)).
        arma::mat M = Bgroup[i] * Bwrap.L;      // Multiply the group product with the current L matrix
        M = M * arma::diagmat(Bwrap.D);        // Apply the diagonal matrix Dwrap

        // Decompose M into new L, D, R using pivoted QR decomposition.
        LDRMatrix M_LDR = qr_LDR(M);

        // Combine the new L, D, R with the current R matrix:
        // Update the L, D, and R components of the wrapped result.
        Bwrap.L = M_LDR.L;               // Update Lwrap
        Bwrap.D = M_LDR.D;               // Update Dwrap
        Bwrap.R = M_LDR.R * Bwrap.R;     // Update Rwrap by multiplying with the existing R
    }

    // Step 4: Return the final wrapped LDRMatrix.
    return Bwrap;
}

// END OF LINEAR ALGEBRA SUBROUTINE //

// MODEL SUBROUTINE //

arma::mat build_Kmat(int L, double t, double mu) {
    // Step 1: Compute the total number of lattice sites (N = L^2).
    //         This assumes the lattice is square with side length L.
    int N = L * L; 

    // Step 2: Initialize the kinetic matrix K as an NxN zero matrix.
    arma::mat K = arma::zeros<arma::mat>(N, N); 

    // Step 3: Iterate over all lattice sites (x, y) to populate the hopping terms.
    for (int x = 0; x < L; ++x) {             // Loop over x-coordinates
        for (int y = 0; y < L; ++y) {         // Loop over y-coordinates
            // Map the 2D lattice index (x, y) to a 1D index (site).
            int site = x * L + y;

            // +x neighbor with periodic boundary conditions.
            // Compute the 1D index of the neighbor to the right (x + 1, y).
            int x_neighbor = ((x + 1) % L) * L + y;

            // Add hopping terms to K for the +x neighbor.
            K(site, x_neighbor) = -t;         // Hopping from site to x_neighbor
            K(x_neighbor, site) = -t;         // Conjugate hopping (symmetric)

            // +y neighbor with periodic boundary conditions.
            // Compute the 1D index of the neighbor above (x, y + 1).
            int y_neighbor = x * L + (y + 1) % L;

            // Add hopping terms to K for the +y neighbor.
            K(site, y_neighbor) = -t;         // Hopping from site to y_neighbor
            K(y_neighbor, site) = -t;         // Conjugate hopping (symmetric)
        }
    }

    // Step 4: Add the chemical potential term (-mu) to the diagonal of K.
    //         This represents the on-site energy for each lattice site.
    for (int site = 0; site < N; site++) {
        K(site, site) = -mu;
    }

    // Step 5: Return the constructed kinetic matrix K.
    return K;
}


arma::mat calculate_exp_Kmat(const arma::mat& K, const double& delta_tau, double sign) {
    // Step 1: Compute the prefactor for the matrix exponential.
    //         The prefactor is: sign * delta_tau * K
    //         - `sign`: Determines the direction of propagation (useful when using symmetrize version).
    //         - `delta_tau`: Time step size for Trotter discretization.
    //         - `K`: The kinetic matrix representing the hopping terms and chemical potential.
    arma::mat prefactor = sign * delta_tau * K;

    // Step 2: Compute the matrix exponential of the prefactor.
    //         The matrix exponential exp(prefactor) is used to describe propagation in time.
    //         arma::expmat computes the matrix exponential.
    arma::mat expK = arma::expmat(prefactor);

    // Step 3: Return the resulting matrix exponential.
    return expK;
}


arma::Mat<int> initialize_random_ising(int L, int L_tau) {
    // Step 1: Compute the total number of lattice sites.
    int N = L * L; // Total number of lattice sites

    // Step 2: Create a matrix to store the Ising configurations.
    arma::Mat<int> config(N, L_tau); // Matrix with N rows and L_tau columns

    // Step 3: Set up a random number generator.
    std::random_device rd;          // Non-deterministic seed
    std::mt19937 gen(rd());         // Mersenne Twister generator seeded with rd()
    std::uniform_int_distribution<> dis(0, 1); // Uniform distribution between 0 and 1

    // Step 4: Populate the Ising configuration matrix with random spins.
    for (int i = 0; i < N; ++i) {          // Loop over lattice sites
        for (int j = 0; j < L_tau; ++j) {  // Loop over time slices
            // Generate a random spin (+1 or -1)
            config(i, j) = (dis(gen) == 0) ? -1 : 1;
        }
    }

    // Step 5: Return the initialized configuration matrix.
    return config;
}

arma::mat calculate_exp_Vmat(double sgn, const double& alpha, arma::Mat<int> s) {
    // Step 1: Get the dimensions of the input matrix `s`.
    //         `rows` is the number of rows, and `cols` is the number of columns.
    arma::uword rows = s.n_rows;
    arma::uword cols = s.n_cols;

    // Step 2: Initialize the output matrix `expV` with the same dimensions as `s`.
    //         This will store the computed values of exp(alpha * sgn * s[i, j]).
    arma::mat expV(rows, cols);

    // Step 3: Convert the integer matrix `s` to a double matrix using arma::conv_to.
    //         This ensures compatibility with floating-point operations.
    arma::mat s_double = arma::conv_to<arma::mat>::from(s);

    // Step 4: Compute the element-wise exponential:
    //         exp(alpha * sgn * s[i, j]) for each element in `s`.
    expV = arma::exp(sgn * alpha * s_double);

    // Step 5: Return the resulting matrix `expV`.
    return expV;
}

std::tuple<double, double> update_ratio_hubbard(double G_ii, double s_il, double alpha, double sgn) {
    // Step 1: Compute delta_.
    //         This is the change in the on-site exponential potential term expV.
    //         The formula is derived from the Hubbard-Stratonovich transformation:
    //         delta_ = exp(sgn * -2.0 * alpha * s_il) - 1.0
    //         - `sgn`: A sign factor indicating spin direction.
    //         - `alpha`: Hubbard-Stratonovich parameter.
    //         - `s_il`: The Ising spin variable (+1 or -1).
    double delta_ = std::exp(sgn * -2.0 * alpha * static_cast<double>(s_il)) - 1.0;

    // Step 2: Compute the determinant ratio.
    //         The ratio is given by:
    //         r = 1 + delta_ * (1 - G_ii)
    //         - `G_ii`: The Green's function at the updated site (diagonal element).
    //         - `delta_`: The change computed in Step 1.
    //         This ratio measures how the determinant of the Green's function matrix changes.
    double ratio = 1.0 + delta_ * (1.0 - G_ii);

    // Step 3: Return the determinant ratio and delta_.
    //         These values are returned as a tuple.
    return std::make_tuple(ratio, delta_);
}

void propagate_equaltime_greens(
    arma::mat& G,         // Green's function matrix (NxN), updated in place
    const arma::mat& expK,      
    arma::mat& expV,
    int l,
    bool is_symmetric,
    bool forward
) {
    // Step 1: Retrieve the propagator matrix for time slice l (B_{l+1}).
    //         Ensure l+1 is valid in your calling code to avoid accessing invalid indices.
    arma::mat B_l = calculate_B_matrix(expK, expV, l, is_symmetric);

    // Step 2: Compute the inverse of B_{l+1}. This is used to propagate G forward.
    arma::mat inv_B_l = inv(B_l);

    // Step 4: Update G by performing the propagation: 
    //         G <- B_{l+1} * G * B_{l+1}^{-1}.
    //         This method assumes you have correctly constructed the B matrices from expK and expV.
    if (forward)
    {
        G = B_l * G * inv_B_l;
    } 
    else
    {
        G = inv_B_l * G * B_l;
    }
}


void symmmetric_warp_greens(
    arma::mat& G,                  // Green's function matrix (NxN), updated in place
    const arma::mat& expK,         // Exponential of the kinetic matrix: exp(-ΔτK) (NxN)
    const arma::mat& inv_expK,     // Inverse of expK: exp(+ΔτK) (NxN)
    bool forward                   // Flag indicating the direction of propagation
) {
    // Step 1: Check the propagation direction.
    if (forward) {
        // Forward propagation (imaginary time increases):
        // Update G as: G <- inv(expK) * G * expK.
        G = inv_expK * G * expK;
    } else {
        // Backward propagation (imaginary time decreases):
        // Update G as: G <- expK * G * inv(expK).
        G = expK * G * inv_expK;
    }
}


// Function to perform the local update on the Green's function matrix
void local_update_greens(
    arma::mat& G,              // Green's function matrix (NxN), updated in place
    arma::mat& expV,           // Diagonal matrix (as a vector) for exp(-Δτ * V)
    double r,                  // Determinant ratio
    double delta,              // Change in on-site energy
    int i,                     // Site index being updated
    int l
) {

    // Step 1: Extract column G(:, i) into u
    arma::vec u = G.col(i);

    // Step 2: Extract row G(i, :) into v and subtract identity row
    arma::vec v = G.row(i).t();  // Transpose to match the vector orientation
    v(i) = v(i) - 1.0;       // Subtract the identity term for v(i)

    // Step 3: Perform the rank-1 update G = G + (Δ/R) * u * v.t()
    G += (delta / r) * (u * v.t());

    // Step 4: Update the diagonal on-site energy expV
    expV(i,l) = expV(i,l) * (1.0 + delta);
}

void sweep_time_slices(
    arma::mat& Gup, arma::mat& Gdn, 
    arma::mat& expVup, arma::mat& expVdn, 
    const arma::mat& expK, const arma::mat& inv_expK, 
    arma::Mat<int>& s, double alpha, 
    int L_tau, int N, int nwrap, int nstab, 
    bool is_symmetric, 
    std::mt19937& rng, std::uniform_real_distribution<double>& dis, 
    double& acceptance_rate
) {

    int accepted = 0;
    for (int l = 0; l < L_tau; l++) {
        // propagate Green function locally
        propagate_equaltime_greens(Gup, expK, expVup, l, is_symmetric, true);
        propagate_equaltime_greens(Gdn, expK, expVdn, l, is_symmetric, true);

        // Symmetric warp if enabled
        if (is_symmetric) {
            symmmetric_warp_greens(Gup, expK, inv_expK, true);
            symmmetric_warp_greens(Gdn, expK, inv_expK, true);
        }

        // Shuffle the sites
        std::vector<int> shuffled_sites = shuffle_numbers(s.n_rows, rng);

        for (int& site : shuffled_sites) {
            // Update ratio for spin-up and spin-down
            double Gup_ii = Gup(site, site);
            double Gdn_ii = Gdn(site, site);
            double s_il = s(site, l);

            auto [ratio_up, delta_up] = update_ratio_hubbard(Gup_ii, s_il, alpha, 1);
            auto [ratio_dn, delta_dn] = update_ratio_hubbard(Gdn_ii, s_il, alpha, -1);

            // Calculate probability
            double prob = std::abs(ratio_up * ratio_dn);

            if (dis(rng) < prob) {
                // Increment accepted counter
                accepted += 1;

                // Flip the Ising configuration
                s(site, l) = -s_il;

                // Update Green's functions locally
                local_update_greens(Gup, expVup, ratio_up, delta_up, site, l);
                local_update_greens(Gdn, expVdn, ratio_dn, delta_dn, site, l);
            }
        }

        // Symmetric warp reverse if enabled
        if (is_symmetric) {
            symmmetric_warp_greens(Gup, expK, inv_expK, false);
            symmmetric_warp_greens(Gdn, expK, inv_expK, false);
        }

        // Propagate Green's functions for the next time slice
        if (l % nstab == 0) {
            // Calculate G(tau, tau) from scratch

            // Shift expVup and expVdn to align with the current slice
            arma::mat expVup_shifted = shiftMatrixColumnsLeft(expVup, l + 1);
            arma::mat expVdn_shifted = shiftMatrixColumnsLeft(expVdn, l + 1);

            // Wrap into Bup and Bdn = B(tau,0)B(beta,0)
            LDRMatrix Bup = wrap_B_matrices(expK, expVup_shifted, nwrap, is_symmetric);
            LDRMatrix Bdn = wrap_B_matrices(expK, expVdn_shifted, nwrap, is_symmetric);

            // Recalculate G_eq
            auto [new_Gup, signdetGup] = calculate_invIpA(Bup);
            auto [new_Gdn, signdetGdn] = calculate_invIpA(Bdn);

            Gup = new_Gup;
            Gdn = new_Gdn;
        }
    }
    
    // Calculate acceptance rate
    acceptance_rate += static_cast<double>(accepted) / N / L_tau;
}

// Measurement functions
double measure_double_occupancy(const arma::mat& Gup, const arma::mat& Gdn) {
    int N = Gup.n_rows; // Number of sites
    double D = 0.0;

    for (int i = 0; i < N; ++i) {
        D += (1.0 - Gup(i, i)) * (1.0 - Gdn(i, i));
    }

    return D / N; // Average double occupancy per site
}

double measure_kinetic_energy(const arma::mat& Gup, const arma::mat& Gdn, double t, int L) {
    int N = L * L; // Total number of sites
    double E_kin = 0.0;

    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            int site = x * L + y;

            // +x neighbor
            int x_neighbor = ((x + 1) % L) * L + y;
            E_kin += -t * (Gup(site, x_neighbor) + Gdn(site, x_neighbor));

            // +y neighbor
            int y_neighbor = x * L + (y + 1) % L;
            E_kin += -t * (Gup(site, y_neighbor) + Gdn(site, y_neighbor));
        }
    }

    return 2.0 * E_kin;
}

double measure_potential_energy(const arma::mat& Gup, const arma::mat& Gdn, double U) {
    int N = Gup.n_rows; // Number of sites
    double E_pot = 0.0;

    for (int i = 0; i < N; ++i) {
        E_pot += U * (1.0 - Gup(i, i)) * (1.0 - Gdn(i, i));
    }

    return E_pot;
}

// Statistics computation function
std::pair<double, double> compute_stats(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / data.size() - mean * mean);
    double stderr = stdev / std::sqrt(data.size());

    return {mean, stderr};
}

// Function to read input parameters from a file
void read_input_parameters(
    const std::string& filename,
    double& t, double& mu, double& U, double& beta, double& delta_tau,
    int& L, int& nwrap, int& nstab, int& nsweep_measure, int& nsweep_thermal, int& nbins,
    bool& is_symmetric, bool& is_forward
) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error: Could not open input file " + filename);
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string key;
        if (line.find('=') != std::string::npos) {
            iss >> key;
            if (key == "t") iss.ignore(3) >> t;
            else if (key == "mu") iss.ignore(3) >> mu;
            else if (key == "U") iss.ignore(3) >> U;
            else if (key == "beta") iss.ignore(3) >> beta;
            else if (key == "delta_tau") iss.ignore(3) >> delta_tau;
            else if (key == "L") iss.ignore(3) >> L;
            else if (key == "nwrap") iss.ignore(3) >> nwrap;
            else if (key == "nstab") iss.ignore(3) >> nstab;
            else if (key == "nsweep_measure") iss.ignore(3) >> nsweep_measure;
            else if (key == "nsweep_thermal") iss.ignore(3) >> nsweep_thermal;
            else if (key == "nbins") iss.ignore(3) >> nbins;
            else if (key == "is_symmetric") iss.ignore(3) >> std::boolalpha >> is_symmetric;
            else if (key == "is_forward") iss.ignore(3) >> std::boolalpha >> is_forward;
        }
    }

    infile.close();
}

// END OF MODEL SUBROUTINE //
