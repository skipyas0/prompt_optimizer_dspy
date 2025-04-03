import utils
import dspy

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
def score_connections(solution, gold, logger=None):
    s = utils.recursive_string_normalize(utils.deseparate_into_lists(solution))
    g = utils.recursive_string_normalize(utils.deseparate_into_lists(gold))
    if s is None:
        logger.warning(f"Solution {s} is invalid")
        return 0.0
    if g is None:
        logger.error(f"Could parse gold answer {s}")
        return 0.0
    return match_lists(s, g)

def match_lists(solution, gold, logger=None):
    print("matching", solution, gold)
    acc = 0.0
    if len(solution) != len(gold):
        return acc
    for s, g in zip(sorted(solution), sorted(gold)):
        if type(s) == list and type(g) == list:
            acc += match_lists(s, g, logger=logger)
        elif s.strip().lower() == g.strip().lower():
            acc += 1.0
    avg = acc / len(gold)
    print("score", avg)
    return 0.0 if avg < 0.25 else avg - 0.25 if avg < 0.9 else 1.0



if __name__ == "__main__":
    s = "excellence: expert, ace, hotshot, gladiator; relationships: soulmate, her, rapture; playfulness: popcorn, crackerjack, joker, chock, jack; mechanics: wrench, tire, rocketry"
    g = "ace, crackerjack, expert, hotshot;chock, jack, tire, wrench;gladiator, her, joker, signs;popcorn, rapture, rocketry, soulmate"
    s = utils.recursive_string_normalize(utils.deseparate_into_lists(s))
    g = utils.recursive_string_normalize(utils.deseparate_into_lists(g))
    print(match_lists(s, g))