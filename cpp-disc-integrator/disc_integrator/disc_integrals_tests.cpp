#include "gtest/gtest.h"
#include <string>
#include <vector>
#include "disc_integrals.h"
#include <tuple>


// Define custom tuple types to be used
using VecTup = std::tuple<std::vector<double>,std::vector<double>,
std::vector<double>,std::vector<double>>;
using ParTup = std::tuple<int, bool>;


// define the fixture used to perform testing
class DiscIntegrationTest : 
    public ::testing::Test,
    public ::testing::WithParamInterface<ParTup> {
    private:
        // initialize the private vector data as part of a struct
        struct myData{
            // define the vector data
            std::vector<double> mu_values = {-1.0, -0.5, 0.0, 0.5, 1.0};
            std::vector<double> ang_fnc = {0.1, 0.5, 0.9, 1.3, 1.7};
            std::vector<double> fir_der = {0.0, 0.75, 0.8, 0.75, 0.7};
            std::vector<double> sec_der = {0.0, 0.0, 0.04, -0.06, -0.08};
            // define the function that retrieves the tuple
            VecTup get_tup(struct myData* self){
                // create the input tuple
                VecTup my_tuple = std::make_tuple(
                    self->mu_values, self->ang_fnc, 
                    self->fir_der, self->sec_der
                );
                // return it
                return my_tuple;
                };
        };
        myData privateData;    
    protected:
        DiscIntegrationTest() {}     
        // store a vector tuple that holds the vector data
        VecTup my_tup = privateData.get_tup(&privateData);
        // DEFINE THE EXPECTED OUTPUT INTEGRATION VALUES
        // NOTE(!): we remove the first two values of the vectors
        // to perform the disc integrations: these need to be removed
        // due to the calculation of the numerical derivatives!
        double expect_fii_trap = 1.63125;
        double expect_fii_simp = 1.466666666666666;
        double expect_sii_trap = 1.2228125;
        double expect_sii_simp = 1.047083333333333;
};


// UTILITY FUNCTIONS


// function used to compute the first disc integral
double get_first_integral(
    int num_threads, bool use_parallel,
    string my_ldf_string, string my_integ_string,
    VecTup my_tup
){
    // unpack the input tuple
    auto [mu_values, ang_fnc, fir_der, sec_der] = my_tup;
    // initialize the object
    DiscIntegration my_obj(
        mu_values.size(), &mu_values[0], &ang_fnc[0], &fir_der[0],
        &sec_der[0], my_ldf_string, my_integ_string,
        num_threads, use_parallel
    );
    // compute the first integral
    double my_result = my_obj.compute_first_disc_integral();   
    // return it
    return my_result;
}


// function used to compute the second disc integral
double get_second_integral(
    int num_threads, bool use_parallel,
    string my_ldf_string, string my_integ_string,
    VecTup my_tup
){
    // unpack the input tuple
    auto [mu_values, ang_fnc, fir_der, sec_der] = my_tup;
    // initialize the object
    DiscIntegration my_obj(
        mu_values.size(), &mu_values[0], &ang_fnc[0], &fir_der[0],
        &sec_der[0], my_ldf_string, my_integ_string,
        num_threads, use_parallel
    );
    // compute the first integral
    double my_result = my_obj.compute_second_disc_integral();   
    // return it
    return my_result;
}


// TEST LIBRARY


TEST_P(DiscIntegrationTest, FirstDITrapzEddington){
    // set up parameters for this test
    auto [num_threads, use_parallel] = GetParam();
    string my_edd = "Eddington";
    string my_trp = "trapz";
    // get the numerical disc integration result
    double my_result = get_first_integral(
        num_threads, use_parallel, my_edd, my_trp,
        my_tup
    );
    // perform the assertion check
    ASSERT_DOUBLE_EQ(expect_fii_trap, my_result);    
}


TEST_P(DiscIntegrationTest, FirstDISimpsonEddington){
    // set up parameters for this test
    auto [num_threads, use_parallel] = GetParam();
    string my_edd = "Eddington";
    string my_trp = "simpson";
    // get the numerical disc integration result
    double my_result = get_first_integral(
        num_threads, use_parallel, my_edd, my_trp,
        my_tup
    );
    // perform the assertion check
    ASSERT_DOUBLE_EQ(expect_fii_simp, my_result);    
}


TEST_P(DiscIntegrationTest, SecondDITrapzEddington){
    // set up parameters for this test
    auto [num_threads, use_parallel] = GetParam();
    string my_edd = "Eddington";
    string my_trp = "trapz";
    // get the numerical disc integration result
    double my_result = get_second_integral(
        num_threads, use_parallel, my_edd, my_trp,
        my_tup
    );
    // perform the assertion check
    ASSERT_DOUBLE_EQ(expect_sii_trap, my_result);    
}


TEST_P(DiscIntegrationTest, SecondDISimpsonEddington){
    // set up parameters for this test
    auto [num_threads, use_parallel] = GetParam();
    string my_edd = "Eddington";
    string my_trp = "simpson";
    // get the numerical disc integration result
    double my_result = get_second_integral(
        num_threads, use_parallel, my_edd, my_trp,
        my_tup
    );
    // perform the assertion check
    ASSERT_DOUBLE_EQ(expect_sii_simp, my_result);    
}


INSTANTIATE_TEST_SUITE_P(
    DiscIntegrationTests,
    DiscIntegrationTest,
    ::testing::Values(
        std::make_tuple(1, false),
        std::make_tuple(2, true)
    )
);
