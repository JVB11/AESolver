/**
 * @file ld_functions.cpp
 * @author Jordan Van Beeck (jordanvanbeeck@hotmail.com)
 * @brief Contains class and methods to compute limb-darkening functions.
 * @version 1.0.0
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "ld_functions.h"

// enum that allows mapping for the limb-darkening functions
enum LDOptions {
  eddington,
};

// unordered map from strings to enum values for limb-darkening functions
unordered_map<string, LDOptions> my_string_mapper = {
    {"eddington", LDOptions::eddington},
};

/**
 * @brief Construct a new LimbDarkeningFunctions :: LimbDarkeningFunctions
 * object, which contains methods used to compute/access the Limb-darkening
 * functions.
 */
LimbDarkeningFunctions::LimbDarkeningFunctions(int arrsiz, double *mus,
                                               double *lds, string selectionvar,
                                               const int num_threads = 4,
                                               bool para = false) {
  // store the number of threads and whether to use parallel implementations
  this->numthreads = num_threads;
  this->paral = para;
  // initialize the function selection integer and muval size
  this->function_selection = my_string_mapper[selectionvar];
  this->muvalsize = arrsiz;
  // initialize the muvals attribute
  this->muvals = mus;
  // initialize the ld functions attribute
  this->ld_funcs = lds;
};

/**
 * @brief Compute the Eddington limb-darkening function (serial implementation).
 */
void LimbDarkeningFunctions::eddington() {
  // fill the data structure with the Eddington limb darkening function values
  for (int k = 0; k < this->muvalsize; k++) {
    this->ld_funcs[k] = 1.0 + (1.5 * this->muvals[k]);
  }
};

/**
 * @brief Compute the Eddington limb-darkening function (parallel
 * implementation).
 */
void LimbDarkeningFunctions::eddington_parallel() {
// fill the data structure with the Eddington limb darkening function values
#pragma omp parallel for num_threads(numthreads) shared(ld_funcs, muvals)
  for (int k = 0; k < this->muvalsize; k++) {
    this->ld_funcs[k] = 1.0 + (1.5 * this->muvals[k]);
  }
};

/**
 * @brief Computes the correct Limb-darkening function based on the passed
 * selection parameter. Defaults to Eddington limb-darkening if the selection
 * parameter is not recognized. Changes between serial and parallel
 * implementations based on the instance variable 'paral' (initialized using
 * input; default: false).
 */
void LimbDarkeningFunctions::compute() {
  switch (this->function_selection) {
  case LDOptions::eddington: {
    if (this->paral) {
      eddington_parallel();
    } else {
      eddington();
    }
  }
  default: {
    if (this->paral) {
      eddington_parallel();
    } else {
      eddington();
    }
  }
  };
};

/**
 * @brief Destroy the LimbDarkeningFunctions :: LimbDarkeningFunctions object.
 */
LimbDarkeningFunctions::~LimbDarkeningFunctions() {}
