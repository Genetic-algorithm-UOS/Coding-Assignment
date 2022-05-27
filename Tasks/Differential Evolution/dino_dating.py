
def dating_fitness(trex: list) -> float:
    """
    This function maps a T-Rex candidate solution to their dating fitness value
    :param trex: A vector representing a candidate solution for a T-Rex
    :return: float, dating fitness value
    """
    # [brain_size, teeth_size, height, weight, camouflage_level, claw_size, aggression] = trex
    goals = [100, 30, 8, 5, 42, 7, 0]
    value = 0
    for x, goal in zip(trex, goals):
        value += 100 / (1 + ((goal-x)**2))

    return value / len(trex)
