from pydantic import BaseModel


class PowInputModel(BaseModel):
    """
    Pydantic model for incoming requests.
        - inputs: JSON string containing the actual inputs for the proof of weights request.
        - signature: Signature of the inputs. Signed by the validator's hotkey.
        - sender: Sender's wallet address.
        - netuid: The originating subnet the request comes from.
    """

    inputs: str
    signature: str
    sender: str
    netuid: int
