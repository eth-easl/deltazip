rm -rf build/
cmake . -DCMAKE_PREFIX_PATH=./lib -Bbuild -DBUILD_GDS_EXAMPLE=ON
cd build && make -j4
mv bin ../