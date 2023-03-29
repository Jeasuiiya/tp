# Build

## Tensorflow

### Debug
Debug工具`tf-run-graph`, 执行tf1计算图
```bash
# 依赖
pip install -r dev-requirements.txt
./scripts/build-tf --config-only
bazel build tf-run-graph --config=dbg
bazel build rpc_server --config=dbg
```

运行`tf-run-graph`
```bash
# 依赖
pip install tensorflow networkx
./bazel-bin/rpc_server localhost:9001
TF_PLACEMENT_RPC_ADDRESS=localhost:9001 ./bazel-bin/tf-run-graph --graph=/path/to/graph --train_op=<trainop>
```

**模型文件**

* [resnet50](https://gist.githubusercontent.com/Yiklek/cc66295cef7361c6a701c9408f1e2661/raw/c7e1bc36f178d57480ae701bc4f7a11cc5b1a530/resnet50.pbtxt)

### Release

打包Tensorflow到dist
```bash
# 依赖
pip install -r dev-requirements.txt
./scripts/build-tf
```