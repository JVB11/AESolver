#!/bin/bash

cd ./disc_integrator/ || exit
cmake -Bcmbuild
cmake --build cmbuild
./cmbuild/a.out
rm -rf ./cmbuild
cd ..
