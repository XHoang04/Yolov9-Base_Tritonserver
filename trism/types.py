import numpy as np

__TRITON_STRING_TO_NUMPY__ = {
  "BOOL": bool,
  "UINT8": np.uint8,
  "UINT16": np.uint16,
  "UINT32": np.uint32,
  "UINT64": np.uint64,
  "INT8": np.int8,
  "INT16": np.int16,
  "INT32": np.int32,
  "INT64": np.int64,
  "FP16": np.float16,
  "FP32": np.float32,
  "FP64": np.float64,
  "STRING": np.object_,
  "BYTES": np.bytes_,
}

__NUMPY_TO_TRITON_STRING__ = {
  bool: "BOOL",
  np.uint8: "UINT8",
  np.uint16: "UINT16",
  np.uint32: "UINT32",
  np.uint64: "UINT64",
  np.int8: "INT8",
  np.int16: "INT16",
  np.int32: "INT32",
  np.int64: "INT64",
  np.float16: "FP16",
  np.float32: "FP32",
  np.float64: "FP64",
  np.object_: "STRING",
  np.bytes_: "BYTES"
  
}

def trt2np(dtype: str) -> np.dtype:
  dtype = dtype.upper()
  if dtype in __TRITON_STRING_TO_NUMPY__:
    return __TRITON_STRING_TO_NUMPY__[dtype]
  elif dtype.startswith("TYPE_"):
    return __TRITON_STRING_TO_NUMPY__[dtype.replace("TYPE_", "")]
  raise TypeError(f"Cannot convert triton type {dtype} to numpy type!")

def np2trt(dtype: np.dtype, prefix: bool = False) -> str:
  dtype = __NUMPY_TO_TRITON_STRING__[dtype]
  return f"TYPE_{dtype}" if prefix else dtype
