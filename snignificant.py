import math
from dataclasses import dataclass


@dataclass
class Snig:
    as_float: float
    as_str: str
    exponent: int
    sig_figs: int


def _round_up_to_multiple(x, multiple_of):
    return x + (-x % multiple_of)


def _round_to_nearest_multiple(x, multiple_of):
    return multiple_of * round(float(x) / multiple_of)

# todo: I'm not sure that the as_str values really make sense for any base other than 10

def round_to_sf(x, sf=1, base=10, round_up=False) -> Snig:
    rounder = _round_up_to_multiple if round_up else _round_to_nearest_multiple
    n = math.floor(math.log(x, base)) - sf + 1
    rounded = rounder(x, base**n)
    m = max(-n, 0)
    
    return Snig(as_float=rounded, as_str=f"{rounded:.{m}f}", exponent=n, sig_figs=sf)

def round_to_position(x, pos: int, base: int = 10) -> Snig:
    rounded = _round_to_nearest_multiple(x, base ** pos)
    m = max(-pos, 0)
    # todo: set sig_figs
    return Snig(as_float=rounded, as_str=f"{rounded:.{m}f}", exponent=pos, sig_figs=None)


if __name__ == "__main__":
    ...