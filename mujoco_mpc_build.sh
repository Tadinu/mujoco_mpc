#https://github.com/google-deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml

# Ubuntu 20.04: clang-12
# Ubuntu 22.04: clang-11/12/13
#sudo apt update
#sudo apt install libc++-18-dev
#sudo apt install libc++abi-18-dev
#sudo apt install libomp-18-dev

mkdir -p build
pushd build

BUILD_TYPE=Release #RelWithDebInfo Debug
cmake .. -G Ninja \
         -DCMAKE_C_COMPILER:STRING=clang-18 -DCMAKE_CXX_COMPILER:STRING=clang++-18 -DMUJOCO_HARDEN:BOOL=ON \
		 -DCMAKE_INSTALL_PREFIX=../release \
		 -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
		 -DMJPC_BUILD_GRPC_SERVICE:BOOL=OFF \
 		 -DSDFLIB_USE_SYSTEM_SPDLOG:BOOL=ON \
                 -Dcasadi_DIR=$MEDIA_EXT_DRIVE/MUJOCO/CASADI/casadi/release/lib/cmake/casadi \
		 -Ddrake_DIR=$MEDIA_EXT_DRIVE/11_MPC/DRAKE_MPC/drake/release/lib/cmake/drake \
                 -DCMAKE_ISPC_COMPILER=$MEDIA_EXT_DRIVE/ISPC/bin/ispc \
                 #-DCMAKE_CXX_FLAGS:STRING="-stdlib=libc++ -D_GLIBCXX_USE_CXX11_ABI=1" \
                 #-DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,--no-as-needed -stdlib=libc++"

cmake --build . -j8 --config=Release
cmake --install .
popd

# Copy mjpc release (libs + headers) to UE's RRSimBase
#source ./copy_mjpcrelease_to_rrsim_base.sh

