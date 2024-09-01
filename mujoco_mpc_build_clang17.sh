#https://github.com/google-deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml

#sudo apt update
#sudo apt install libc++-17-dev
#sudo apt install libc++abi-17-dev
mkdir -p build
pushd build

cmake .. -G Ninja \
      -DCMAKE_C_COMPILER:STRING=clang-17 -DCMAKE_CXX_COMPILER:STRING=clang++-17 \
                  #-DMUJOCO_HARDEN:BOOL=ON \
		  -DCMAKE_INSTALL_PREFIX=../release \
		  -DCMAKE_CXX_STANDARD=17 \
		  -DCMAKE_CXX_FLAGS:STRING="-stdlib=libc++" \
		  -DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,--no-as-needed -stdlib=libc++" \
		  -DCMAKE_BUILD_TYPE:STRING=Release \
		  -DMJPC_BUILD_GRPC_SERVICE:BOOL=OFF \
		  -Dcasadi_DIR=../../CASADI/casadi/release/lib/cmake/casadi \
		  -DCMAKE_ISPC_COMPILER=/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/ISPC/bin/ispc

cmake --build . -j8
cmake --install .
popd

# Copy mjpc release (libs + headers) to UE's RRSimBase
#source ./copy_mjpcrelease_to_rrsim_base.sh

