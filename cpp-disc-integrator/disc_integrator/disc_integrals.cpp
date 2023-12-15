/**
 * @file disc_integrals.cpp
 * @author Jordan Van Beeck (jordanvanbeeck@hotmail.com)
 * @brief Contains class and methods to compute disc integrals.
 * @version 1.0.0
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "disc_integrals.h"

/**
 * @brief Construct a new DiscIntegration :: DiscIntegration object,
 * which contains methods to compute the disc integral factors
 * described in Van Beeck et al. (forthcoming).
 */
DiscIntegration::DiscIntegration(int orig_size, double *mus,
                                 double *angular_function, double *first_mu_der,
                                 double *second_mu_der,
                                 string ld_function_string,
                                 string integrationstring, int num_threads,
                                 bool para) {
  // store the number of threads and whether to use parallel implementations
  this->numthreads = num_threads;
  this->paral = para;
  // store the original data structure size
  this->origsize = orig_size;
  // store the pointers to the original data structures
  this->mymus = mus;
  this->myang = angular_function;
  this->firstder = first_mu_der;
  this->secondder = second_mu_der;
  /* Adjust the data to the appropriate length,
  creating instance variables in the process */
  adjust_data_to_integration_length();
  // store the vector size
  this->vecsize = (int)this->mymus_b.size();
  // allocate memory for the LD function
  this->ldfunc.resize(this->vecsize);
  // Compute the LD function.
  compute_ld_function(ld_function_string);
  // Obtain the mapping of the string to the switch variable
  this->integration_string = integrationstring;
};

/**
 * @brief Constructs the data vectors of the correct length to compute the
 * integral factors.
 */
void DiscIntegration::adjust_data_to_integration_length() {
  // loops over all mu values in the integration length and
  // adds the necessary values to the (new) integration vectors
  for (int i = 0; i < this->origsize; i++) {
    if ((this->mymus[i] >= 0.0) && (this->mymus[i] <= 1.0)) {
      this->mymus_b.push_back(this->mymus[i]);
      this->myang_b.push_back(this->myang[i]);
      this->firstder_b.push_back(this->firstder[i]);
      this->secondder_b.push_back(this->secondder[i]);
    }
  }
};

/**
 * @brief Computes the LD function and stores the result.
 */
void DiscIntegration::compute_ld_function(string ldfunction_string) {
  // initialize the LD-function-computing object
  LimbDarkeningFunctions my_ld(this->vecsize, this->mymus_b.data(),
                               this->ldfunc.data(), ldfunction_string,
                               this->numthreads, this->paral);
  // compute the LD function values
  my_ld.compute();
};

/**
 * @brief Perform integration based on selection string.
 */
double DiscIntegration::perform_integration(vector<double> &my_integrand) {
  // create a numerical integrator object
  NumericalDoubleInputIntegrator my_num_inter(
      this->vecsize, my_integrand.data(), this->mymus_b.data(), true,
      this->integration_string, this->numthreads, this->paral);
  // switch statement between the different integrations
  return my_num_inter.integrate();
};

/**
 * @brief Computes the first disc integral factor in a serial or parallel way.
 */
double DiscIntegration::compute_first_disc_integral() {
  // choose between serial or parallel implementation based on stored boolean
  if (this->paral) {
    return parallel_first();
  } else {
    return serial_first();
  };
};

/**
 * @brief Computes the first disc integral factor (serial implementation).
 */
double DiscIntegration::serial_first() {
  // initialize the vector for the integrand for the first disc integral
  vector<double> first_integrand(this->vecsize);
  // fill the vector with values
  for (int j = 0; j < this->vecsize; j++) {
    first_integrand[j] = this->mymus_b[j] * this->myang_b[j] * this->ldfunc[j];
  }
  // return the integration result
  return perform_integration(first_integrand);
}

/**
 * @brief Computes the first disc integral factor (parallel implementation).
 */
double DiscIntegration::parallel_first() {
  // initialize the vector for the integrand for the first disc integral
  vector<double> first_integrand(this->vecsize);
// fill the vector with values
#pragma omp parallel for num_threads(numthreads)                               \
    shared(first_integrand, mymus_b, myang_b, ldfunc)
  for (int j = 0; j < this->vecsize; j++) {
    first_integrand[j] = this->mymus_b[j] * this->myang_b[j] * this->ldfunc[j];
  }
  // return the integration result
  return perform_integration(first_integrand);
}

/**
 * @brief Computes the second disc integral factor in a serial or parallel way.
 */
double DiscIntegration::compute_second_disc_integral() {
  // choose between serial or parallel implementation based on stored boolean
  if (this->paral) {
    return parallel_second();
  } else {
    return serial_second();
  };
};

/**
 * @brief Computes the second disc integral factor (serial implementation).
 */
double DiscIntegration::serial_second() {
  // initialize the vector for the integrand for the second disc integral
  vector<double> second_integrand(this->vecsize);
  // fill the vector with values
  for (int j = 0; j < this->vecsize; j++) {
    double mmu = this->mymus_b[j]; // reduces number of lookups
    second_integrand[j] = ((2.0 * mmu * this->firstder_b[j]) +
                           ((pow(mmu, 2.0) - 1.0) * this->secondder_b[j])) *
                          this->ldfunc[j] * mmu;
  }
  // return the integration result
  return perform_integration(second_integrand);
};

/**
 * @brief Computes the second disc integral factor (parallel implementation).
 */
double DiscIntegration::parallel_second() {
  // initialize the vector for the integrand for the second disc integral
  vector<double> second_integrand(this->vecsize);
// fill the vector with values
#pragma omp parallel for num_threads(numthreads)                               \
    shared(second_integrand, mymus_b, firstder_b, secondder_b, ldfunc)
  for (int j = 0; j < this->vecsize; j++) {
    double mmu = this->mymus_b[j]; // reduces number of lookups
    second_integrand[j] = ((2.0 * mmu * this->firstder_b[j]) +
                           ((pow(mmu, 2.0) - 1.0) * this->secondder_b[j])) *
                          this->ldfunc[j] * mmu;
  }
  // return the integration result
  return perform_integration(second_integrand);
};

/**
 * @brief Destroy the DiscIntegration :: DiscIntegration object.
 */
DiscIntegration::~DiscIntegration() {}
