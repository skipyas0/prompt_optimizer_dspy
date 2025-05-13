import utils

def exact_match_float(solution, gold):
    """
    Grading function for sequences (and other int/float output benchmarks)
    """
    g = float(gold)
    try:
        solution = float(solution)
        grade = 1.0 if solution==g else 0.0
    except (ValueError, TypeError):
        grade = 0.0
    return grade

def match_lists(solution, gold):
    """
    Grading function for connections.
    Returns proportion of correct groupings in solution.
    """
    if not isinstance(solution, str) or not isinstance(gold, str):
        return 0.0
    
    # normalize both
    s = sorted(utils.sep_norm_sort(solution))
    g = sorted(utils.sep_norm_sort(gold))

    score = sum([int(''.join(_s) == ''.join(_g)) for _s,_g in zip(s,g)]) / len(g)
    return score

def test_code(solution, gold):
    """
    Grading function for codecontests, checks if solution code solves test cases.
    Gives credit for each passed case and partial credit for successful execution.
    """
    gold_inputs, gold_outputs, _ = gold
    score = 0
    for gold_input, gold_output in zip(gold_inputs, gold_outputs):
        out = utils.execute_code(solution, gold_input)
        if not isinstance(out, str) or "Exception:" in out:
            # execution failed (score += 0)
            continue
        elif out == gold_output:
            score += 1
        else:
            score += 0.1 # partial credit: successful execution but wrong output
    return score / len(gold_inputs)
