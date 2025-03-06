import numpy as np
from trism import client


class TritonModel:

  @property
  def model(self) -> str:
    return self._model
  
  @property
  def version(self) -> str:
    return self._version
  
  @property
  def url(self) -> str:
    return self._url
  
  @property
  def grpc(self) -> bool:
    return self._grpc
  
  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs

  def __init__(self, model: str, version: int, url: str, grpc: bool = True) -> None:
    self._url = url
    self._grpc = grpc
    self._model = model
    self._version = str(version) if version > 0 else ""
    self._protoclient = client.protoclient(self.grpc)
    self._serverclient = client.serverclient(self.url, self.grpc)
    self._inputs, self._outputs = client.inout(self._serverclient, self.model, self.version)
  
  def run(self, data: list[np.array]):
    inputs = [self.inputs[i].make_input(self._protoclient, data[i]) for i in range(len(self.inputs))]
    outputs = [output.make_output(self._protoclient) for output in self.outputs]
    results = self._serverclient.infer(self.model, inputs, self.version, outputs)
    return {output.name: results.as_numpy(output.name) for output in self.outputs}
