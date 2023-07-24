mkdir -p builds
mkdir -p builds/debug
cd builds/debug
cmake -DCMAKE_BUILD_TYPE=Debug ../..
make
cd ../..
