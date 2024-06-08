#https://github.com/google-deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml

mkdir -p build
pushd build

cmake .. -G Ninja \
	 -DCMAKE_INSTALL_PREFIX=$HOME/1_MUJOCO/mujoco_mpc/release \
	 -DCMAKE_BUILD_TYPE:STRING=Release \
	 -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON 
cmake --build . -j8 --config=Release
cmake --install .
popd
