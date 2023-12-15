/**
 * @file disc_integrals.h
 * @author Jordan Van Beeck (jordanvanbeeck@hotmail.com)
 * @brief Header file for C++ module that computes disc integral factors (e.g. Burkart, 2012 and Fuller, 2017).
 * @version 1.0.0
 * @date 2022-05-26
 *
 * @copyright Copyright (c) 2022
 *
 */
/* Header file for disc integration class */
// guard
#ifndef DISC_INTEGRALS_H
#define DISC_INTEGRALS_H
// include
#include "ld_functions.h"
#include "num_integration.h"
#include "omp.h"
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>
// namespace
using namespace std;

// class definition
/**
 * @brief Class containing methods used to compute the disc integral factors
 * described in Van Beeck et al. (forthcoming).
 */
class DiscIntegration {
private:
  /**
   * @name Class attributes
   * Contains the DiscIntegration class attributes.
   */
  //@{
  /** Boolean value deciding whether to do parallel integration */
  bool paral;
  /** Number of threads used during parallel integrations */
  int numthreads;
  /** Pointer to data structure holding mu = cos (theta) values*/
  double *mymus;
  /** Vector containing LD function values */
  vector<double> ldfunc;
  /** Pointers to data structures containing values of the angular function
  and its first and second mu derivative */
  double *myang;
  double *firstder;
  double *secondder;
  /** Vectors containing Angular functions, derivatives and integration
  variables that fulfill the boundaries of the angular integration */
  vector<double> mymus_b, myang_b, firstder_b, secondder_b;
  /** Vector that will hold the LD function */
  vector<double> ldf;
  /** String denoting the numerical integration technique used */
  string integration_string;
  /** Integer denoting the size of the vectors that fulfill integration
   * boundaries */
  int vecsize;
  /** Integer denoting the size of the original data structures */
  int origsize;
  //@}
  /**
   * @name Internal class methods
   * Contains the class methods used to perform the disc integrations.
   */
  //@{
  /**Constructs the data vectors of the correct length to compute
  the integral factors.*/
  void adjust_data_to_integration_length();
  /** Computes the LD function and stores the result. */
  void compute_ld_function(string ldfunction_string);
  /** Performs the numerical integration and returns the result. */
  double perform_integration(vector<double> &integrand);
  /** Serial and parallel integration implementation for first integral */
  double serial_first();
  double parallel_first();
  /** Serial and parallel integration implementation for second integral */
  double serial_second();
  double parallel_second();
  //@}
public:
  /**
   * @brief Construct a new DiscIntegration object.
   *
   * @param mus The pointer to the data structure holding values of mu =
   * cos(theta)
   * @param angular_function The pointer to the data structure containing
   * the values of the angular function
   * @param first_mu_der The pointer to the data structure storing values
   * of the first mu derivative of the angular function.
   * @param second_mu_der The pointer to the data structure storing values
   * of the second mu derivative of the angular function.
   * @param ld_function_string String selecting which limb-darkening function is
   * to be used.
   * @param integration_string String selecting which numerical integration
   * technique is to be used.
   */
  DiscIntegration(int orig_size, double *mus, double *angular_function,
                  double *first_mu_der, double *second_mu_der,
                  string ld_function_string, string integration_string,
                  int num_threads = 4, bool para = false);
  /**
   * @brief Computes the first disc integral factor.
   *
   * @return double
   */
  double compute_first_disc_integral();
  /**
   * @brief Computes the second disc integral factor.
   *
   * @return double
   */
  double compute_second_disc_integral();
  /**
   * @brief Destroy the DiscIntegration object.
   */
  ~DiscIntegration();
};
#endif
