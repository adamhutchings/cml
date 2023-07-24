mkdir -p builds
mkdir -p builds/release
cd builds/release
cmake -DCMAKE_BUILD_TYPE=Release ../..
make
cd ../..
