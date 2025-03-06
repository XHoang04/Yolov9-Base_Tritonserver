import numpy as np
from trism import types
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class Inout:

  @property
  def name(self) -> str:
    return self.__name
  
  @property
  def shape(self) -> tuple:
    return self.__shape
  
  @property
  def dtype(self) -> np.dtype:
    return self.__dtype

  def __init__(self, name: str, shape: tuple, dtype: str, *args, **kwds):
    self.__name = name
    self.__shape = shape
    self.__dtype = types.trt2np(dtype)

  def make_input(self, client, data: np.array) -> grpcclient.InferInput | httpclient.InferInput:
    infer = client.InferInput(
      name=self.name,
      shape=data.shape,
      datatype=types.np2trt(self.dtype)
    )
    infer.set_data_from_numpy(data.astype(self.dtype))
    return infer
  
  def make_output(self, client) -> grpcclient.InferRequestedOutput | httpclient.InferRequestedOutput:
    return client.InferRequestedOutput(name=self.name)
