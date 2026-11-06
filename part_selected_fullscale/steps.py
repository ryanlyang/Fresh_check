"""Original screen step arithmetic with the FULL recipe's LR function.

The screen schedule dispatch expects short-recipe fields; binding a private
function namespace avoids changing it for any existing campaign.
"""
import types
from scripts.run_part_ev_screen import training_step as screen_step
from part_ev_fullscale import protocol as full_protocol

training_step = types.FunctionType(screen_step.__code__, dict(screen_step.__globals__, P=full_protocol),
                                   screen_step.__name__, screen_step.__defaults__, screen_step.__closure__)
training_step.__kwdefaults__ = dict(screen_step.__kwdefaults__)
