import utils

float_signature = "question: str -> answer: float"

def exact_match_float(solution, gold, logger=None):
    g = float(gold)
    try:
        solution = float(solution)
        grade = 1.0 if solution==g else 0.0
    except ValueError:
        if logger:
            logger.warning(f"Couldn't convert '{solution}' to float")
        grade = 0.0
    return grade

str_signature = "question: str -> answer: str"


def match_lists(solution, gold, logger=None):
    s = sorted(utils.sep_norm_sort(solution))
    g = sorted(utils.sep_norm_sort(gold))
    score = sum([int(''.join(_s) == ''.join(_g)) for _s,_g in zip(s,g)]) / len(g)
    return score


if __name__ == "__main__":
    s = "expert, ace, hotshot, gladiator; soulmate, her, rapture; popcorn, crackerjack, joker, chock, jack; wrench, tire, rocketry"
    g = "ace, gladiator, expert, hotshot;chock, jack, tire, wrench;gladiator, her, joker, signs;popcorn, rapture, rocketry, soulmate"

    print(match_lists(s, g))