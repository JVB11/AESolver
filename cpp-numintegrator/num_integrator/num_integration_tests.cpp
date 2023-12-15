#include "num_integration.h"
#include "gtest/gtest.h"
#include <string>
#include <tuple>
#include <vector>

// define custom tuple type to be used
using VecTup = std::tuple<std::vector<double>, std::vector<double>>;
using ParTup = std::tuple<int, bool, bool>;

// define the fixture class used for testing
class NumericalDoubleInputIntegratorTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<ParTup> {
private:
  struct myData {
    std::vector<double> integrand = {1., 2., 3., 4., 5.};
    std::vector<double> variable = {0.1, 0.5, 0.9, 1.3, 1.7};
    VecTup get_tup(struct myData *self) {
      VecTup my_tuple = std::make_tuple(self->integrand, self->variable);
      return my_tuple;
    };
  };
  myData privateData;

protected:
  NumericalDoubleInputIntegratorTest() {}
  // store the data vector tuple
  VecTup DataVecs = privateData.get_tup(&privateData);
  // store the results of the integration tests
  double trapz_result = 4.8;
  double simpson_result = 4.8;
};

// TEST LIBRARY

TEST_P(NumericalDoubleInputIntegratorTest, Trapz) {
  // set up parameters for this test
  auto [num_threads, use_parallel, equal_spacing] = GetParam();
  auto [integrand, variable] = DataVecs;
  string my_trapz = "trapz";
  // initialize the object
  NumericalDoubleInputIntegrator my_obj(integrand.size(), &integrand[0],
                                        &variable[0], equal_spacing, my_trapz,
                                        num_threads, use_parallel);
  // get the numerical integration result
  double my_result = my_obj.integrate();
  // perform the check
  ASSERT_DOUBLE_EQ(trapz_result, my_result);
}

TEST_P(NumericalDoubleInputIntegratorTest, Simpson) {
  // set up parameters for this test
  auto [num_threads, use_parallel, equal_spacing] = GetParam();
  auto [integrand, variable] = DataVecs;
  string my_trapz = "simpson";
  // initialize the object
  NumericalDoubleInputIntegrator my_obj(integrand.size(), &integrand[0],
                                        &variable[0], equal_spacing, my_trapz,
                                        num_threads, use_parallel);
  // get the numerical integration result
  double my_result = my_obj.integrate();
  // perform the check
  ASSERT_DOUBLE_EQ(simpson_result, my_result);
}

INSTANTIATE_TEST_SUITE_P(
    NumericalIntegrationTests, NumericalDoubleInputIntegratorTest,
    ::testing::Values(
        std::make_tuple(1, false, true),  // first value = num_threads
        std::make_tuple(1, false, false), // second value = use_parallel
        std::make_tuple(2, true, true),   // third value = equal_spacing
        std::make_tuple(2, true, false)));
