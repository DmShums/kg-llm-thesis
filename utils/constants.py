import logging
from typing import Optional, List, Union
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class BinaryOutputFormat(BaseModel):
    answer: bool


class BinaryOutputFormatWithReasoning(BaseModel):
    reasoning: str
    answer: bool


class TokensUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMCallOutput(BaseModel):
    message: Union[str, bool]
    usage: TokensUsage
    logprobs: Optional[List] = None
    parsed: Optional[BaseModel] = None


PAIRS_SEPARATOR = "|"
GT_COL_DIVIDER = "\t"
RESULTS_SEPARATOR = ","

class RepairPlan(BaseModel):
    plan_id: int
    score: float
    reason: str


from pydantic import RootModel

class RepairRankingOutput(RootModel[List[RepairPlan]]):
    pass