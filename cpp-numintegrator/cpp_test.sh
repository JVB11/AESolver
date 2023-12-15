#!/bin/bash

cd ./num_integrator/ || exit
cmake -Bcmbuild
cmake --build cmbuild
./cmbuild/a.out
rm -rf ./cmbuild
cd ..
