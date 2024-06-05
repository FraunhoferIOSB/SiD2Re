import numpy as np
import random
from sid2re.driftgenerator.concept.nodes.root_distributions import UniformDistribution, PeriodicalDistribution, \
    ConstantDistribution, GaussianDistribution


def test_distributions_faulty_sensor_behaviour():
    for dist_class in [UniformDistribution, PeriodicalDistribution, ConstantDistribution, GaussianDistribution]:
        value = ConstantDistribution.generate(w=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), timestamp=0,
                                              minimum=0, maximum=1)
        assert value == 0.0


def test_constant():
    random.seed(42)
    np.random.seed(42)
    value = ConstantDistribution.generate(w=np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]), timestamp=0,
                                          minimum=0, maximum=1)
    assert value == 0.5
    value = ConstantDistribution.generate(w=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), timestamp=0,
                                          minimum=1, maximum=1)
    assert value == 1.0
    value = ConstantDistribution.generate(w=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), timestamp=0,
                                          minimum=-1, maximum=-1)
    assert value == -1.0
    value = ConstantDistribution.generate(w=np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]), timestamp=0,
                                          minimum=-1, maximum=1)
    assert value == 0.0


def test_periodical():
    random.seed(42)
    np.random.seed(42)
    epsilon = 0.1
    value = [PeriodicalDistribution.generate(w=np.array([0.1, 0.5, 1, 0.5, 1, 0, 0, 0, 0, 0]),
                                             timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    assert (0.0 - epsilon <= np.mean(value) <= 0.0 + epsilon)
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))
    value = [PeriodicalDistribution.generate(w=np.array([0.4, 0, 1, 0, 1, 0, 0, 0, 0, 0]),
                                             timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    assert (0.0 - epsilon <= np.mean(value) <= 0.0 + epsilon)
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))
    value = [PeriodicalDistribution.generate(w=np.array([0.7, 0, 1, 0, 1, 0, 0, 0, 0, 0]),
                                             timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    print(np.mean(value))
    assert (0.0 <= np.mean(value))
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))
    value = [PeriodicalDistribution.generate(w=np.array([0.7, 0, 1, 0, 1, 1, 0, 0, 0, 0]),
                                             timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    print(np.mean(value))
    assert (np.mean(value) <= 0.0)
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))


def test_gaussian():
    random.seed(42)
    np.random.seed(42)
    epsilon = 0.1
    value = [GaussianDistribution.generate(w=np.array([0.6, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]),
                                           timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    assert (1.0 - epsilon <= np.mean(value) <= 1.0 + epsilon)
    assert (0.1 - epsilon <= np.var(value) <= 0.1 + epsilon)
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))
    value = [GaussianDistribution.generate(w=np.array([0.6, 10, 0, 0, 0, 0, 0, 0, 0, 0]),
                                           timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))


def test_uniform():
    random.seed(42)
    np.random.seed(42)
    epsilon = 0.1
    value = [UniformDistribution.generate(w=np.array([0.6, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]),
                                          timestamp=i, minimum=-5, maximum=5) for i in np.arange(10000)]
    assert (1.0 - epsilon <= np.mean(value) <= 1.0 + epsilon)
    assert (np.amax(value) <= 5.0)
    assert (-5.0 <= np.amin(value))
