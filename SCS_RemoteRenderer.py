import ray

from SCS_Renderer import *


@ray.remote
class SCS_RemoteRenderer(SCS_Renderer):

    def __init__(self, remote_storage=None):

        super().__init__(remote_storage=remote_storage)