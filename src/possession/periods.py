"""Run independent prediction blocks without exposing reference labels or masks."""


def predict_periods(observations, blocks, new_predictor):
    """Return H/A/D/X labels, with fresh state per half-open observation block.

    Blocks come from published periods or observable discontinuities, never GT.
    new_predictor() returns a callable taking one observation block and returning
    its labels. It must create independent state each time. Frames outside blocks
    are X. No football inference or ground-truth logic is implemented here.
    """
    blocks = list(blocks)
    previous_stop = 0
    for start, stop in blocks:
        if (type(start) is not int or type(stop) is not int
                or not previous_stop <= start < stop <= len(observations)):
            raise ValueError("blocks must be ordered, disjoint, nonempty and within observations")
        previous_stop = stop

    labels = ["X"] * len(observations)
    for start, stop in blocks:
        predict = new_predictor()
        result = list(predict(observations[start:stop]))
        if len(result) != stop - start or any(label not in ("H", "A", "D", "X") for label in result):
            raise ValueError("predictor must return one H/A/D/X label per input frame")
        labels[start:stop] = result
    return labels
