#include <iostream>
#include "gtest/gtest.h"


int main(int argc, char**argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "RUNNING TESTS...\n";
    int ret{RUN_ALL_TESTS()};
    if (!ret)
        std::cout << "<<<SUCCESS>>>\n";
    else
        std::cout << "FAILED\n";
    return 0;
}
