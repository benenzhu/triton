declare -x PATH="/A/code/triton_learn/pyenv:/bin/usr/local/nvm/versions/node/v16.20.2/bin:/root/.vscode-server-insiders/cli/servers/Insiders-336db9ece67f682159078ea1b54212de7636d88a/server/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-12.0/bin"
#export LLVM_BUILD_DIR=/A/__code/triton_learn/llvm-project/build

#export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include 
#export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib 
#export LLVM_SYSPATH=$LLVM_BUILD_DIR 
export TRITON_BUILD_WITH_CLANG_LLD=1
source /A/code/triton_learn/pyenv/bin/activate
export DEBUG=1
export MAX_JOBS=150
# export TRITON_OFFLINE_BUILD=1
pip install -e python

