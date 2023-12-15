/**
 * @file ld_funcs.h
 * @author Jordan Van Beeck (jordanvanbeeck@hotmail.com)
 * @brief Header file for C++ module that computes limb-darkening functions.
 * @version 1.0.0
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */
/* Header file for limb darkening class */
// guard
#ifndef LD_FUNCTIONS_H
#define LD_FUNCTIONS_H
// include
#include "omp.h"
#include <string>
#include <unordered_map>
// namespace
using namespace std;

// class definition
class LimbDarkeningFunctions {
private:
  /**
   * @name Private class attributes:
   * The mu = cos(theta) values and function selection attribute.
   */
  //@{
  /** Boolean value deciding whether to do parallel computation */
  bool paral;
  /** Number of threads used during parallel computation */
  int numthreads;
  /** The mu = cos(theta) values */
  double *muvals;
  /** The selection int, translated using an enum */
  int function_selection;
  /** The size of the muvals data structure */
  int muvalsize;
  /** The pointer to the data structure that will hold the limb-darkening
  functions */
  double *ld_funcs;
  //@}
  /**
   * @brief Private class functions:
   * The methods used to compute the limb darkening functions.
   */
  //@{
  /** Eddington limb darkening function */
  void eddington();
  void eddington_parallel();
  //@}
public:
  /**
   * @brief Construct the LimbDarkeningFunctions object.
   *
   * @param arrsize The size of the data structure.
   * @param mus The pointer to the data structure storing
   * mu = cos(theta) values.
   * @param lds The pointer to the data structure that will store the
   * limb-darkening function values.
   * @param selectionvar The string selection variable
   * (used to select LD function).
   */
  LimbDarkeningFunctions(int arrsize, double *mus, double *lds,
                         string selectionvar, const int num_threads, bool para);
  /**
   * @brief Compute the limb-darkening function based on
   * the supplied pointer to the data structure of mu = cos(theta) values
   * and a case/switch variable. Uses Eddington limb darkening by default.
   */
  void compute();
  /**
   * @brief Destroy the LimbDarkeningFunctions object.
   */
  ~LimbDarkeningFunctions();
};

#endif
