/**
 * @file numerical_integration.cpp
 * @author Jordan Van Beeck (jordanvanbeeck@hotmail.com)
 * @brief Contains class and methods to perform numerical integrations.
 * @version 1.0.0
 * @date 2022-05-26
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "num_integration.h"

// enum that allows mapping for the integration functions
enum IntegrationOptions {
  trapz,
  simpson,
};

// unordered map from strings to enum values for integration functions
unordered_map<string, IntegrationOptions> my_integration_mapper = {
    {"trapz", IntegrationOptions::trapz},
    {"simpson", IntegrationOptions::simpson},
};

/**
 * @brief Construct a new Numerical Integration :: Numerical Integration object,
 * an object of the class that contains methods to perform
 * numerical integration based on input arrays of the integrand
 * and the variable of integration.
 *
 * @param integrand pointer to data structure storing integrand/integral kernel
 * @param var pointer to data structure storing variable of integration
 * @param es boolean that decides whether to use equally spaced or unequally
 * spaced (spacing of integration variable grid) numerical integration methods
 */
NumericalDoubleInputIntegrator::NumericalDoubleInputIntegrator(
    int arrsize, double *integrand, double *var, bool es,
    string integration_string, int num_threads, bool para) {
  // initialize the boolean attributes
  this->equally_spaced = es;
  this->paral = para;
  // initialize the number of threads attribute
  this->numthreads = num_threads;
  // initialize the array size attribute
  this->arr_size = arrsize;
  // initialize the pointer attributes
  this->my_var = var;
  this->my_integrand = integrand;
  // obtain the mapping of the string to the switch variable
  this->integration_switch = my_integration_mapper[integration_string];
};

/**
 * @brief Performs numerical integration using the trapezoid scheme, and inputs.
 *
 */
double NumericalDoubleInputIntegrator::trapz() {
  // differentiate between equally spaced integration and non-equally spaced
  if (equally_spaced) {
    // initialize the increment variable
    double increment = my_var[1] - my_var[0];
    // initialize the sum result variable
    double sum_result =
        ((my_integrand[0] + my_integrand[this->arr_size - 1]) / 2.0);
    // add the sum of all but the first and last elements of the integrand
    for (int i = 1; i < this->arr_size - 1; i++) {
      sum_result += my_integrand[i];
    }
    // return the integration result
    return sum_result * increment;
  } else {
    // initialize local variables
    double local_integrand_sum, local_increment, integration_result = 0.0;
    // update the integration result using the non-equal spaced grid
    for (int i = 1; i < this->arr_size; i++) {
      local_increment = my_var[i] - my_var[i - 1];
      local_integrand_sum = (my_integrand[i] + my_integrand[i - 1]) / 2.0;
      integration_result += local_integrand_sum * local_increment;
    }
    // return the integration result
    return integration_result;
  }
};

/**
 * @brief Parallelized implementation of the trapz integration.
 *
 */
double NumericalDoubleInputIntegrator::trapz_parallel() {
  // differentiate between equally spaced integration and non-equally spaced
  if (equally_spaced) {
    // initialize the increment variable
    double increment = my_var[1] - my_var[0];
    // initialize the sum result variable
    double sum_result =
        ((my_integrand[0] + my_integrand[this->arr_size - 1]) / 2.0);
// add the sum of all but the first and last elements of the integrand
#pragma omp parallel for num_threads(numthreads) shared(my_integrand)          \
    reduction(+ : sum_result)
    for (int i = 1; i < this->arr_size - 1; i++) {
      sum_result += my_integrand[i];
    }
    // return the integration result
    return sum_result * increment;
  } else {
    // initialize local variables
    double local_integrand_sum, local_increment, integration_result = 0.0;
// update the integration result using the non-equal spaced grid
#pragma omp parallel for num_threads(numthreads) shared(my_var, my_integrand)  \
    reduction(+ : integration_result)
    for (int i = 1; i < this->arr_size; i++) {
      local_increment = my_var[i] - my_var[i - 1];
      local_integrand_sum = (my_integrand[i] + my_integrand[i - 1]) / 2.0;
      integration_result += local_integrand_sum * local_increment;
    }
    // return the integration result
    return integration_result;
  }
};

/**
 * @brief Performs numerical integration using Simpson's 1/3 rule, and inputs.
 * Returns trapezoidal rule integration result for a unequally spaced
 * integration variable grid.
 */
double NumericalDoubleInputIntegrator::simpson() {
  // equally spaced: return result of Simpson's 1/3 rule
  if (equally_spaced) {
    // check if the array size is odd, otherwise return trapz result
    if (((this->arr_size - 1) % 2)) {
      // even array size, return trapz result
      return trapz();
    } else {
      // initialize increment
      double increment = my_var[1] - my_var[0];
      // initialize the half size local variable
      int half_size = this->arr_size / 2;
      int n = this->arr_size - 1;
      // initialize local sum variable using integral bounds
      double sum_result = my_integrand[0] + my_integrand[n];
      // add (all but one) odd-numbered and (all) even-numbered elements
      for (int i = 1; i < half_size; i++) {
        sum_result += 4.0 * my_integrand[2 * i - 1] + 2.0 * my_integrand[2 * i];
      }
      // add last odd-numbered element
      sum_result += 4.0 * my_integrand[n - 1];
      // return the integration result
      return (increment / 3.0) * sum_result;
    }
  } else {
    // unequally spaced: return trapz result
    return trapz();
  }
};

/**
 * @brief Parallelized implementation of the integration using Simpson's 1/3
 * rule.
 */
double NumericalDoubleInputIntegrator::simpson_parallel() {
  // equally spaced: return result of Simpson's 1/3 rule
  if (equally_spaced) {
    // check if the array size is odd, otherwise return trapz result
    if ((this->arr_size - 1) % 2) {
      // even array size, return trapz result
      return trapz();
    } else {
      // initialize increment
      double increment = my_var[1] - my_var[0];
      // initialize the half size local variable
      int half_size = this->arr_size / 2;
      int n = this->arr_size - 1;
      // initialize local sum variable using integral bounds
      double sum_result = my_integrand[0] + my_integrand[n];
// add (all but one) odd-numbered and (all) even-numbered elements
#pragma omp parallel for num_threads(numthreads) shared(my_integrand)          \
    reduction(+ : sum_result)
      for (int i = 1; i < half_size; i++) {
        sum_result += 4.0 * my_integrand[2 * i - 1] + 2.0 * my_integrand[2 * i];
      }
      // add last odd-numbered element
      sum_result += 4.0 * my_integrand[n - 1];
      // return the integration result
      return (increment / 3.0) * sum_result;
    }
  } else {
    // unequally spaced: return trapz result
    return trapz();
  }
};

/**
 * @brief Perform integration based on selection string.
 */
double NumericalDoubleInputIntegrator::integrate() {
  // switch statement between the different integrations
  switch (this->integration_switch) {
  case IntegrationOptions::trapz: {
    return trapz();
  }
  case IntegrationOptions::simpson: {
    return simpson();
  }
  default: {
    return trapz();
  }
  }
  return 0.0;
};

/**
 * @brief Destroy the
 * NumericalDoubleInputIntegrator::NumericalDoubleInputIntegrator object.
 */
NumericalDoubleInputIntegrator::~NumericalDoubleInputIntegrator() {}
