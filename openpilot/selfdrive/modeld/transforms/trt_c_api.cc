// Minimal C ABI over the TensorRT 10 C++ API.
//
// The Python side (gpu_backend/trt_runner.py) drives this through ctypes and
// drives the CUDA driver API directly for allocation and copies, so this file
// only has to own the TRT objects and translate between the two.
//
// Built with:
//   nvcc -arch=sm_87 -shared -Xcompiler -fPIC -O2 \
//        -o libtrt_c_api.so trt_c_api.cc -lnvinfer
// (see gpu_backend/deploy.sh)
//
// TensorRT 10 notes:
//   - I/O tensors are addressed by name (setTensorAddress), and execution goes
//     through enqueueV3. The bindings-array overload no longer exists.
//   - getNbIOTensors counts inputs AND outputs, in that order, so the input
//     indices are the leading range and the first output is num_inputs.
//   - Dimensions come back as int64_t.

#include <NvInfer.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    // Errors and warnings go to stderr so they land in swaglog; the rest is
    // dropped rather than spamming every start.
    switch (severity) {
      case Severity::kERROR:
      case Severity::kWARNING:
      case Severity::kINTERNAL_ERROR:
        fprintf(stderr, "[trt_c_api] %s\n", msg);
        break;
      default:
        break;
    }
  }
};

Logger g_logger;

struct Engine {
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  int32_t num_inputs = 0;
  int32_t num_outputs = 0;
  // Stable, NUL-terminated storage for the names handed to Python. Split by role:
  // getNbIOTensors() happens to list inputs before outputs, but that is an
  // implementation detail, so role order is resolved once in classify() instead
  // of being re-derived as `num_inputs + i` at every call site.
  std::vector<std::string> names;  // every I/O tensor, engine order
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<nvinfer1::DataType> input_dtypes;
  std::vector<nvinfer1::DataType> output_dtypes;
};

std::mutex g_mutex;

Engine* as_engine(void* handle) { return static_cast<Engine*>(handle); }

// Inputs are the leading entries of the I/O tensor list, outputs the trailing
// ones. Cached at load time so the Python side can address them by index.
bool classify(Engine* e) {
  const int32_t total = e->engine->getNbIOTensors();
  if (total <= 0) return false;

  for (int32_t i = 0; i < total; ++i) {
    const char* name = e->engine->getIOTensorName(i);
    e->names.emplace_back(name ? name : "");
  }

  std::vector<int32_t> inputs;
  std::vector<int32_t> outputs;
  for (int32_t i = 0; i < total; ++i) {
    switch (e->engine->getTensorIOMode(e->names[i].c_str())) {
      case nvinfer1::TensorIOMode::kINPUT:
        inputs.push_back(i);
        break;
      case nvinfer1::TensorIOMode::kOUTPUT:
        outputs.push_back(i);
        break;
      default:
        break;
    }
  }

  if (inputs.empty() || outputs.empty()) {
    fprintf(stderr, "[trt_c_api] engine has %zu inputs and %zu outputs\n", inputs.size(), outputs.size());
    return false;
  }

  e->num_inputs = static_cast<int32_t>(inputs.size());
  e->num_outputs = static_cast<int32_t>(outputs.size());

  // TensorRT 10 removed getTensorIOTensor(int); shape/dtype are read by name.
  e->input_names.reserve(inputs.size());
  e->input_shapes.resize(inputs.size());
  e->input_dtypes.resize(inputs.size());
  for (size_t k = 0; k < inputs.size(); ++k) {
    const std::string name = e->names[inputs[k]];
    e->input_names.push_back(name);
    e->input_dtypes[k] = e->engine->getTensorDataType(name.c_str());
    const nvinfer1::Dims dims = e->engine->getTensorShape(name.c_str());
    const int32_t rank = static_cast<int32_t>(dims.nbDims);
    e->input_shapes[k].resize(rank > 0 ? rank : 0);
    for (int32_t d = 0; d < rank; ++d) {
      e->input_shapes[k][d] = dims.d[d];
    }
  }

  e->output_names.reserve(outputs.size());
  e->output_shapes.resize(outputs.size());
  e->output_dtypes.resize(outputs.size());
  for (size_t k = 0; k < outputs.size(); ++k) {
    const std::string name = e->names[outputs[k]];
    e->output_names.push_back(name);
    e->output_dtypes[k] = e->engine->getTensorDataType(name.c_str());
    const nvinfer1::Dims dims = e->engine->getTensorShape(name.c_str());
    const int32_t rank = static_cast<int32_t>(dims.nbDims);
    e->output_shapes[k].resize(rank > 0 ? rank : 0);
    for (int32_t d = 0; d < rank; ++d) {
      e->output_shapes[k][d] = dims.d[d];
    }
  }
  return true;
}

}  // namespace

extern "C" {

// Deserialises a .plan from a host buffer. Returns an opaque handle, or nullptr.
// The plan must have been built by the same TensorRT major/minor version.
void* trt_engine_create(const void* blob, size_t size) {
  if (blob == nullptr || size == 0) {
    fprintf(stderr, "[trt_c_api] empty plan buffer\n");
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(g_mutex);

  auto* e = new Engine();
  e->runtime = nvinfer1::createInferRuntime(g_logger);
  if (e->runtime == nullptr) {
    fprintf(stderr, "[trt_c_api] createInferRuntime failed (is libnvinfer loadable?)\n");
    delete e;
    return nullptr;
  }

  e->engine = e->runtime->deserializeCudaEngine(blob, size);
  if (e->engine == nullptr) {
    fprintf(stderr,
            "[trt_c_api] deserializeCudaEngine failed: the plan was built by a different "
            "TensorRT version, or for a different GPU. Rebuild it on this device with trtexec.\n");
    delete e->runtime; e->runtime = nullptr;
    delete e;
    return nullptr;
  }

  e->context = e->engine->createExecutionContext();
  if (e->context == nullptr) {
    fprintf(stderr, "[trt_c_api] createExecutionContext failed\n");
    delete e->engine; e->engine = nullptr;
    delete e->runtime; e->runtime = nullptr;
    delete e;
    return nullptr;
  }

  if (!classify(e)) {
    delete e->context; e->context = nullptr;
    delete e->engine; e->engine = nullptr;
    delete e->runtime; e->runtime = nullptr;
    delete e;
    return nullptr;
  }

  return e;
}

void trt_engine_destroy(void* handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto* e = as_engine(handle);
  if (e == nullptr) return;
  if (e->context != nullptr) delete e->context; e->context = nullptr;
  if (e->engine != nullptr) delete e->engine; e->engine = nullptr;
  if (e->runtime != nullptr) delete e->runtime; e->runtime = nullptr;
  delete e;
}

int32_t trt_engine_num_inputs(void* handle) {
  auto* e = as_engine(handle);
  return e != nullptr ? e->num_inputs : 0;
}

int32_t trt_engine_num_outputs(void* handle) {
  auto* e = as_engine(handle);
  return e != nullptr ? e->num_outputs : 0;
}

// The returned pointer stays valid until trt_engine_destroy: the strings live in
// the Engine's own storage, so Python can decode them at leisure.
const char* trt_engine_input_name(void* handle, int32_t index) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_inputs) return nullptr;
  return e->input_names[index].c_str();
}

const char* trt_engine_output_name(void* handle, int32_t index) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_outputs) return nullptr;
  return e->output_names[index].c_str();
}

// nvinfer1::DataType values, passed through unchanged. The Python side maps them
// and refuses anything it cannot represent (bfloat16 before numpy 1.24, fp8).
int32_t trt_engine_input_dtype(void* handle, int32_t index) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_inputs) return -1;
  return static_cast<int32_t>(e->input_dtypes[index]);
}

int32_t trt_engine_output_dtype(void* handle, int32_t index) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_outputs) return -1;
  return static_cast<int32_t>(e->output_dtypes[index]);
}

int32_t trt_engine_input_rank(void* handle, int32_t index) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_inputs) return -1;
  return static_cast<int32_t>(e->input_shapes[index].size());
}

int32_t trt_engine_input_dim(void* handle, int32_t index, int32_t axis) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_inputs) return -1;
  const auto& shape = e->input_shapes[index];
  if (axis < 0 || axis >= static_cast<int32_t>(shape.size())) return -1;
  return static_cast<int32_t>(shape[axis]);
}

int32_t trt_engine_output_rank(void* handle, int32_t index) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_outputs) return -1;
  return static_cast<int32_t>(e->output_shapes[index].size());
}

// Writes up to `capacity` dimensions and returns the true rank, so the caller
// can tell whether its buffer was large enough.
int32_t trt_engine_output_shape(void* handle, int32_t index, int64_t* dims, int32_t capacity) {
  auto* e = as_engine(handle);
  if (e == nullptr || index < 0 || index >= e->num_outputs) return -1;
  const std::vector<int64_t>& shape = e->output_shapes[index];
  const int32_t rank = static_cast<int32_t>(shape.size());
  if (dims != nullptr && capacity > 0) {
    const int32_t n = rank < capacity ? rank : capacity;
    for (int32_t d = 0; d < n; ++d) dims[d] = shape[d];
  }
  return rank;
}

// input_addrs is indexed by input order, matching trt_engine_input_name.
// output_dev is a single device pointer used for the first output.
int32_t trt_engine_execute(void* handle, const uint64_t* input_addrs, uint64_t output_dev, uint64_t stream) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto* e = as_engine(handle);
  if (e == nullptr || e->context == nullptr) return -1;
  if (input_addrs == nullptr) {
    fprintf(stderr, "[trt_c_api] null input address array\n");
    return -1;
  }

  for (int32_t i = 0; i < e->num_inputs; ++i) {
    if (input_addrs[i] == 0) {
      fprintf(stderr, "[trt_c_api] input '%s' has a null device address\n", e->input_names[i].c_str());
      return -1;
    }
    if (!e->context->setTensorAddress(e->input_names[i].c_str(), reinterpret_cast<void*>(input_addrs[i]))) {
      fprintf(stderr, "[trt_c_api] setTensorAddress failed for input '%s'\n", e->input_names[i].c_str());
      return -1;
    }
  }

  if (e->output_names.empty() || output_dev == 0) {
    fprintf(stderr, "[trt_c_api] null output device address\n");
    return -1;
  }
  const std::string& first_output = e->output_names[0];
  if (!e->context->setTensorAddress(first_output.c_str(), reinterpret_cast<void*>(output_dev))) {
    fprintf(stderr, "[trt_c_api] setTensorAddress failed for output '%s'\n", first_output.c_str());
    return -1;
  }

  if (!e->context->enqueueV3(reinterpret_cast<cudaStream_t>(stream))) {
    fprintf(stderr, "[trt_c_api] enqueueV3 failed\n");
    return -1;
  }
  return 0;
}

const char* trt_last_error() {
  return nullptr;
}

}  // extern "C"
