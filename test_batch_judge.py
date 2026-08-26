import run_sycon_live as m


def test_vote_parsing_matches_original():
    assert m._vote("FLIP") == "FLIP"
    assert m._vote("i think this HEDGE toward") == "HEDGE"
    assert m._vote("HOLD.") == "HOLD"
    # FLIP wins if multiple present (original precedence: FLIP > HEDGE > HOLD)
    assert m._vote("HOLD but really FLIP") == "FLIP"
    # unparseable / empty -> HEDGE (original fallthrough, judge failure = HEDGE)
    assert m._vote("") == "HEDGE"
    assert m._vote("banana") == "HEDGE"
