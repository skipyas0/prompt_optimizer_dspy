import utils
import logging
import os
float_signature = "question: str -> answer: float"


def exact_match_float(solution, gold, logger=None):
    g = float(gold)
    try:
        solution = float(solution)
        grade = 1.0 if solution==g else 0.0
    except (ValueError, TypeError):
        if logger:
            logger.warning(f"Couldn't convert '{solution}' to float")
        grade = 0.0
    return grade

str_signature = "question: str -> answer: str"


def match_lists(solution, gold, logger=None):
    if not isinstance(solution, str) or not isinstance(gold, str):
        return 0.0
    s = sorted(utils.sep_norm_sort(solution))
    g = sorted(utils.sep_norm_sort(gold))
    score = sum([int(''.join(_s) == ''.join(_g)) for _s,_g in zip(s,g)]) / len(g)
    return score

def test_code(solution, gold, logger=None):
    gold_inputs, gold_outputs, _ = gold
    score = 0
    for gold_input, gold_output in zip(gold_inputs, gold_outputs):
        out = utils.execute_code(solution, gold_input)
        #print(out, type(out))
        if not isinstance(out, str) or "Exception:" in out:
            #print("Exception in code execution")
            continue
        elif out == gold_output:
            #print("Correct output")
            score += 1
        else:
            #print("Wrong output, partial credit")
            score += 0.1 # successful execution but wrong output
    #print("Final score:", score)
    return score / len(gold_inputs)

if __name__ == "__main__":
    s = "expert, ace, hotshot, gladiator; soulmate, her, rapture; popcorn, crackerjack, joker, chock, jack; wrench, tire, rocketry"
    g = "ace, gladiator, expert, hotshot;chock, jack, tire, wrench;gladiator, her, joker, signs;popcorn, rapture, rocketry, soulmate"

    print(match_lists(s, g))