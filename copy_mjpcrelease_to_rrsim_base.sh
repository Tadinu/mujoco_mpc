DEST_RELEASE_DIR=${1:-"/home/tad/0_UE/UE5_FLAPTTER/Plugins/RRSimBase/ThirdParty/mujoco_mpc/release"}

# Headers
DEST_INCL_DIR="$DEST_RELEASE_DIR/include/"
rm -rf $DEST_INCL_DIR
mkdir -p $DEST_INCL_DIR
find ./mjpc -name '*.h' -exec cp --parents \{\} /$DEST_INCL_DIR \;
rm $DEST_INCL_DIR/mjpc/app.h
rm $DEST_INCL_DIR/mjpc/simulate.h
rm -rf $DEST_INCL_DIR/mjpc/grpc

# Libs
DEST_LIB_DIR="$DEST_RELEASE_DIR/lib/"
rm -rf $DEST_LIB_DIR
mkdir -p $DEST_LIB_DIR
cp -f build/lib/libmjpc.a $DEST_LIB_DIR
cp -f build/lib/liblqr.a $DEST_LIB_DIR
cp -f build/lib/libthreadpool.a $DEST_LIB_DIR
#cp -f build/lib/libccd.a $DEST_LIB_DIR 
ls -al $DEST_LIB_DIR

# ABSEIL-CPP
# NOTE: vim absl/random/internal/seed_material.cc & disable ABSL_RANDOM_USE_GET_ENTROPY
ABSEIL_DEST_RELEASE_DIR="/home/tad/0_UE/UE5_FLAPTTER/Plugins/RRSimBase/ThirdParty/abseil-cpp/release"
ABSEIL_DEST_INCL_DIR="$ABSEIL_DEST_RELEASE_DIR/include/"
ABSEIL_DEST_LIB_DIR="$ABSEIL_DEST_RELEASE_DIR/lib/"
mkdir -p $ABSEIL_DEST_INCL_DIR
mkdir -p $ABSEIL_DEST_LIB_DIR

pushd build/_deps/abseil-cpp-src
find absl -name '*.h' -exec cp --parents \{\} /$ABSEIL_DEST_INCL_DIR \;
find absl -name '*.inc' -exec cp --parents \{\} /$ABSEIL_DEST_INCL_DIR \;
popd
pushd build/lib
find . -name 'libabsl*.a' -exec cp --parents \{\} /$ABSEIL_DEST_LIB_DIR \;
find . -name 'libabsl*.so' -exec cp --parents \{\} /$ABSEIL_DEST_LIB_DIR \;
#find . -name 'libabsl*.so.*' -exec cp --parents \{\} /$ABSEIL_DEST_LIB_DIR \;
popd

# GRPC
GRPC_DEST_RELEASE_DIR="/home/tad/0_UE/UE5_FLAPTTER/Plugins/RRSimBase/ThirdParty/grpc/release"
GRPC_DEST_INCL_DIR="$GRPC_DEST_RELEASE_DIR/include/"
GRPC_DEST_LIB_DIR="$GRPC_DEST_RELEASE_DIR/lib/"
mkdir -p $GRPC_DEST_INCL_DIR
mkdir -p $GRPC_DEST_LIB_DIR

pushd build/_deps/grpc-src/include
find . -name '*.h' -exec cp --parents \{\} /$GRPC_DEST_INCL_DIR \;
find . -name '*.inc' -exec cp --parents \{\} /$GRPC_DEST_INCL_DIR \;
popd
pushd build/lib
find . -name 'libgrpc*.a' -exec cp --parents \{\} /$GRPC_DEST_LIB_DIR \;
find . -name 'libgrpc*.so' -exec cp --parents \{\} /$GRPC_DEST_LIB_DIR \;
#find . -name 'libgrpc*.so.*' -exec cp --parents \{\} /$GRPC_DEST_LIB_DIR \;
popd
