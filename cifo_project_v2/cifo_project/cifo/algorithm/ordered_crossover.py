def crossover(self, mum, dad):
    """Implements ordered crossover"""

    size = len(mum.vertices)

    # Choose random start/end position for crossover
    alice, bob = [-1] * size, [-1] * size
    start, end = sorted([random.randrange(size) for _ in range(2)])

    # Replicate mum's sequence for alice, dad's sequence for bob
    for i in range(start, end + 1):
        alice[i] = mum.vertices[i]
        bob[i] = dad.vertices[i]

    # # Fill the remaining position with the other parents' entries
    # current_dad_position, current_mum_position = 0, 0
    #
    # for i in chain(range(start), range(end + 1, size)):
    #
    #     while dad.vertices[current_dad_position] in alice:
    #         current_dad_position += 1
    #
    #     while mum.vertices[current_mum_position] in bob:
    #         current_mum_position += 1
    #
    #     alice[i] = dad.vertices[current_dad_position]
    #     bob[i] = mum.vertices[current_mum_position]
    #
    # # Return twins
    # return graph.Tour(self.g, alice), graph.Tour(self.g, bob)
    return mum, dad