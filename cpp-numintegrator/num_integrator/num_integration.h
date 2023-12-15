/**
 * @file numerical_integration.h
 * @author Jordan Van Beeck (jordanvanbeeck@hotmail.com)
 * @brief Header file for C++ module that performs numerical integrations.
 * @version 1.0.0
 * @date 2022-05-26
 *
 * @copyright Copyright (c) 2022
 *
 */
/* Header file for numerical integration class */
// guard
#ifndef NUMERICAL_INTEGRATION_H
#define NUMERICAL_INTEGRATION_H
// include
#include "omp.h"
#include <string>
#include <unordered_map>
// namespace
using namespace std;

// class definition
/**
 * @brief Class containing methodology used to perform numerical integrations
 * based on input values: the integrand/integral kernel and the variable of
 * integration.
 */
class NumericalDoubleInputIntegrator {
private:
  /**
   * @name Class attributes
   * Contains the NumericalDoubleInputIntegrator class attributes.
   */
  //@{
  /** Boolean value deciding whether to do parallel integration */
  bool paral;
  /** Number of threads used during parallel integrations */
  int numthreads;
  /** Size of data structures used */
  int arr_size;
  /** Pointer to data structure storing integrand/integral kernel */
  double *my_integrand;
  /** Pointer to data structure storing variable of integration */
  double *my_var;
  /** Boolean variable that keeps track of whether the integration variable
  is equally spaced */
  bool equally_spaced;
  /** Integer denoting the numerical integration technique used */
  int integration_switch;
  //@}
  /**
   * @name Internal integration methods
   * Contains the numerical integration methods implemented for this class.
   */
  /** @brief Perform numerical integration using the trapezoid scheme. */
  double trapz();
  double trapz_parallel();
  /** @brief Perform numerical integration using Simpson's rule. */
  double simpson();
  double simpson_parallel();

public:
  /**
   * @brief Construct a new NumericalDoubleInputIntegrator object.
   *
   * @param integrand pointer to data structure storing the integrand/integral
   * kernel
   * @param var pointer to data structure storing the variable of integration
   * @param bool boolean stating whether the variable of integration is equally
   * spaced
   */
  NumericalDoubleInputIntegrator(int arrsize, double *integrand, double *var,
                                 bool es, string integration_string,
                                 int num_threads = 4, bool para = false);
  /**
   * @brief Performs the numerical integration.
   */
  double integrate();
  /**
   * @brief Destroy the NumericalDoubleInputIntegrator object.
   */
  ~NumericalDoubleInputIntegrator();
};
#endif
