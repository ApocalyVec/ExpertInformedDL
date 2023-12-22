

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def reverse_tuple(t):
    if len(t) == 0:
        return t
    else:
        return(t[-1],)+reverse_tuple(t[:-1])
