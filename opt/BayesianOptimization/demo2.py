from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

# Let's start by definying our function, bounds, and instanciating an optimization object.

# def black_box_function(x, y):
#     return -x ** 2 - (y - 1) ** 2 + 1
def black_box_function(x, name):
    if name == 1:
        return -x ** 2
    else: return -x ** 3

optimizer = BayesianOptimization(
    f=None,
    # pbounds={'x': (-2, 2), 'y': (-3, 3)},
    pbounds={'x': (-2, 2), 'name': [1, 2]},
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)
target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)
optimizer.register(
    params=next_point_to_probe,
    target=target,
)

for _ in range(10):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)

    print(target, next_point)
print(optimizer.max)

