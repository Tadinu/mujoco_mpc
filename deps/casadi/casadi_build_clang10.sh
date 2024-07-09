# Ubuntu 20.04: clang-10
# Ubuntu 22.04: clang-11/12/13
#sudo apt update
#sudo apt install libc++-dev
#sudo apt install libc++abi-dev
#sudo apt install coinor-libipopt-dev
mkdir -p build
pushd build
export CC=clang-10
export CXX=clang++-10
export FC=gfortran-11
~/CMAKE/3.28.3/bin/cmake .. -DCMAKE_C_COMPILER:STRING=clang-10 -DCMAKE_CXX_COMPILER:STRING=clang++-10 \
		 -DCMAKE_INSTALL_PREFIX=../release \
		 -DCMAKE_CXX_FLAGS:STRING="-stdlib=libc++ -D_GLIBCXX_USE_CXX11_ABI=1" \
		 -DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,--no-as-needed -stdlib=libc++" \
		 -DWITH_PYTHON=OFF -DWITH_IPOPT=ON -DWITH_OPENMP=ON \
		 -DWITH_THREAD=ON -DWITH_WERROR=ON -DWITH_EXTRA_WARNINGS=ON \
		 -DWITH_EXAMPLES=OFF -DBUILD_SHARED_LIBS:BOOL=ON
cmake --build . -j8
cmake --install .
popd
