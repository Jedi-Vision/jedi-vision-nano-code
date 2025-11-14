class InvalidTorchDeviceException(Exception):
    """
    Exception raised for invalid Torch device specifications.
    """

    def __init__(self, device: str, message: str = "Invalid Torch device specified"):
        self.device = device
        self.message = f"{message}: '{device}'"
        super().__init__(self.message)
