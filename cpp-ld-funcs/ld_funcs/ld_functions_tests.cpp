#include "ld_functions.h"
#include "gtest/gtest.h"
#include <string>
#include <tuple>
#include <vector>

// define a custom tuple type
using VecTup = std::tuple<std::vector<double>, std::vector<double>>;
using ParTup = std::tuple<int, bool>;

// define the test fixture
class LimbDarkeningFunctionsTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<ParTup> {
private:
  class myData {
  private:
    // define the vector data
    std::vector<double> mu_values = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::vector<double> limb_darkening_function_values;

  public:
    // constructor
    myData() {
      std::fill_n(std::back_inserter(this->limb_darkening_function_values), 5,
                  0.0);
    };
    // define the member method that retrieves the VecTup
    VecTup get_tup() {
      return std::make_tuple(this->mu_values,
                             this->limb_darkening_function_values);
    };
  };
  myData privateData;

protected:
  LimbDarkeningFunctionsTest() {}
  // store the vector tuple
  VecTup my_tup = privateData.get_tup();
  // store the expected result vector
  std::vector<double> ldf_result_eddington = {-0.5, 0.25, 1.0, 1.75, 2.5};
};

// TEST LIBRARY

TEST_P(LimbDarkeningFunctionsTest, Eddington) {
  // set up the parameters for this test
  auto [num_threads, use_parallel] = GetParam();
  auto [mu_values, limb_darkening_function_values] = my_tup;
  string my_eddington = "Eddington";
  // initialize the object
  LimbDarkeningFunctions my_obj(mu_values.size(), &mu_values[0],
                                &limb_darkening_function_values[0],
                                my_eddington, num_threads, use_parallel);
  // compute the limb-darkening function
  my_obj.compute();
  // assert the elements of the vectors are OK
  ASSERT_EQ(limb_darkening_function_values, ldf_result_eddington);
}

INSTANTIATE_TEST_SUITE_P(LimbDarkeningFunctionsTests,
                         LimbDarkeningFunctionsTest,
                         ::testing::Values(std::make_tuple(1, false),
                                           std::make_tuple(2, true)));
