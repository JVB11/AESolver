#!/bin/bash

cd ./ld_funcs/ || exit
cmake -Bcmbuild
cmake --build cmbuild
./cmbuild/a.out
rm -rf ./cmbuild
cd ..
