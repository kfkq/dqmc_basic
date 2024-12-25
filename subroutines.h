#ifndef SUBROUTINES_H
#define SUBROUTINES_H

#include <stdexcept>
#include <vector>
#include <armadillo>

// Struct for LDR representation
struct LDRMatrix {
    arma::mat L;   // Left matrix 
    arma::vec D;   // Diagonal elements as a vector
    arma::mat R;   // Right matrix
};

// Function declarations
template <typename T>
int sign(T value);

LDRMatrix qr_LDR(arma::mat& M);

std::pair<arma::mat, double> calculate_invIpA(LDRMatrix& A_LDR);

LDRMatrix wrap_B_matrices(std::vector<arma::mat>& B_matrices, int Nwrap);

arma::mat build_Kmat(int L, double t, double mu);

arma::mat calculate_exp_Kmat(const arma::mat& K, const double& delta_tau, double sign);

arma::Mat<int> initialize_random_ising(int L, int L_tau);

arma::mat calculate_exp_Vmat(double sgn, const double& alpha, arma::Mat<int> s);

arma::mat calculate_B_matrix(const arma::mat& expK, arma::mat expV, int time_slice, bool is_symmetric);

std::tuple<double, double> update_ratio_hubbard(double G_ii, double s_il, double alpha, double sgn);

void propagate_equaltime_greens(
    arma::mat& G,
    std::vector<arma::mat>& B,
    arma::mat expK,
    arma::mat expV,
    int l
);

void symmmetric_warp_greens(
    arma::mat& G,
    const arma::mat& expK,
    const arma::mat& inv_expK,
    bool forward
);

void local_update_greens(
    arma::mat& G,
    arma::mat& expV,
    double r,
    double delta,
    int i,
    int l
);

#endif // SUBROUTINES_H
