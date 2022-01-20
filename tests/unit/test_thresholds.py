from ruleskit import Thresholds


class Dummy:
    def __init__(self, **kwargs):
        self.crit_max = 2
        for key in kwargs:
            setattr(self, key, kwargs[key])


def test_thresholds():
    ts = Thresholds("tests/unit/data/thresholds.json")
    dum = Dummy(coverage=0.03)
    assert not ts("coverage", dum)
    assert ts("coverage", Dummy(coverage=0.06))
    assert ts("zscore", Dummy(zscore=2))
    assert ts("zscore", Dummy(zscore=-2))
    assert not ts("zscore", Dummy(zscore=1.5))
    assert not ts("zscore", Dummy(zscore=-1.5))
    assert not ts("crit_mean", Dummy(crit_mean=-1.5))
    assert ts("crit_mean", Dummy(crit_mean=1.5))
    assert ts("crit_min", Dummy(crit_min=1))
    assert ts("crit_min", Dummy(crit_min=-1))
    assert not ts("crit_min", Dummy(crit_min=2.1))
    assert not ts("crit_min", Dummy(crit_min=-2.1))