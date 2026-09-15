"""The typed grammar: construction-time type-checking and a type-directed
enumerator/sampler.

This is the POC's headline lever, productionized. The POC hardcoded one
skeleton (`split -> select -> select -> combine -> paint`) and searched its
args. Here the skeleton is not hardcoded — it *emerges* from the types: a step
may only reference slots that are already filled, so `combine` is unconstructable
before two `select`s, `paint` before a `combine`, and so on. Different task
families induce different legal skeletons from the same rules, which is how one
grammar reaches past the set-op cluster (SLICES.md V5).

`Program(steps)` type-checks at construction (a bad reference raises
`TypeError`). `sample_program` / `enumerate_step` only ever emit type-valid
compositions — the structure V6's GP/PPO will consume as an action mask.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

from object_env.state import ObjState
from object_env.types import ArgType, Grid


@dataclass(frozen=True)
class Param:
    """One typed argument slot: a parameter domain the action draws from
    (COLOR / SETOP / AXIS / DIRECTION / SIZE — never a dataflow type).

    `structural=True` marks a param that steers which slots the action
    reads/writes (the `slot`/`region` selectors) — so it is part of the
    *skeleton*, fixed before the value-arg (color/op/axis/...) search. Value
    params are what a chosen skeleton's arg-search fills in."""

    name: str
    type: ArgType
    domain: tuple  # legal values, for the enumerator
    structural: bool = False


@dataclass(frozen=True)
class Action:
    """A typed verb. `reads`/`writes`/`clears` map the (decoded) args to the
    named slots touched — the symbolic footprint the type-checker walks —
    while `fn(state, **args) -> ObjState | None` is the real executor (None =
    precondition unmet = no-op, the shipped Q7 convention)."""

    name: str
    params: tuple
    fn: Callable
    reads: Callable  # (args: dict) -> tuple[str, ...]  required-filled slots
    writes: Callable  # (args: dict) -> tuple[str, ...]  slots filled after
    clears: Callable = field(default=lambda a: ())  # slots emptied after


@dataclass(frozen=True)
class Step:
    action: Action
    args: dict


class GrammarError(TypeError):
    """A composition that references an unfilled/wrong slot (V5 integration
    test: rejected at construction time)."""


def _advance(filled: frozenset, step: Step) -> frozenset:
    return (filled | set(step.action.writes(step.args))) - set(step.action.clears(step.args))


def type_check(steps, initial=frozenset({"grid"})) -> None:
    """Walk the symbolic type-env; raise `GrammarError` on the first step that
    reads a slot not currently filled."""
    filled = initial
    for i, step in enumerate(steps):
        missing = set(step.action.reads(step.args)) - filled
        if missing:
            raise GrammarError(
                f"step {i} ({step.action.name}) reads unfilled slot(s) "
                f"{sorted(missing)}; filled={sorted(filled)}"
            )
        for p in step.action.params:
            if p.name not in step.args:
                raise GrammarError(f"step {i} ({step.action.name}) missing arg {p.name!r}")
        filled = _advance(filled, step)


class Program:
    """A type-checked, ordered list of steps. Construction is the type gate."""

    def __init__(self, steps):
        self.steps = list(steps)
        type_check(self.steps)

    def run(self, grid: Grid) -> Grid:
        state = ObjState(grid=grid)
        for step in self.steps:
            new = step.action.fn(state, **step.args)
            if new is not None:
                state = new
        return state.grid

    def __len__(self) -> int:
        return len(self.steps)


# ---- the type-directed search surface (V5 fork 3: enumerator ships now) ----
def legal_steps(filled: frozenset, actions) -> list:
    """Every type-valid `Step` from `filled`: each action crossed with every
    param binding whose `reads` are satisfied. The enumerator's one primitive —
    it never emits an ill-typed step."""
    out = []
    for action in actions:
        domains = [p.domain for p in action.params]
        for combo in product(*domains) if domains else [()]:
            args = {p.name: v for p, v in zip(action.params, combo)}
            if set(action.reads(args)) <= filled:
                out.append(Step(action, args))
    return out


def sample_step(rng, actions, filled: frozenset, tries: int = 16):
    """One type-valid `Step` from `filled`, by cheap rejection sampling — pick
    an action, draw its args, accept if its `reads` are satisfied. Falls back to
    a full `legal_steps` enumeration only if rejection keeps missing (so it
    still returns a step whenever any exists). Kept off the cartesian-product
    hot path so the search loop stays fast."""
    for _ in range(tries):
        action = rng.choice(actions)
        args = {p.name: rng.choice(p.domain) for p in action.params}
        if set(action.reads(args)) <= filled:
            return Step(action, args)
    choices = legal_steps(filled, actions)
    return rng.choice(choices) if choices else None


def skeleton_of(program) -> list:
    """The type-valid skeleton underneath a concrete program: its actions with
    only the structural args kept (value args dropped, to be re-searched). Used
    to prove that GIVEN the grammar's skeleton, arg-search rediscovers a task's
    solution — the POC's method, now over a grammar-derived skeleton."""
    return [
        (step.action, {p.name: step.args[p.name] for p in step.action.params if p.structural})
        for step in program.steps
    ]


def _structural_bindings(action):
    """Every combination of an action's *structural* params (the skeleton-level
    ones); a single empty binding when it has none."""
    sparams = [p for p in action.params if p.structural]
    if not sparams:
        return [{}]
    return [dict(zip((p.name for p in sparams), combo)) for combo in product(*(p.domain for p in sparams))]


def enumerate_skeletons(actions, max_depth: int, initial=frozenset({"grid"})):
    """Yield type-valid *skeletons* — lists of `(action, structural_args)` whose
    read/write footprint type-checks — shortest first, up to `max_depth`. The
    skeleton is emergent from the types (not hardcoded as the POC did); the
    value args (colors/op/axis/...) are searched separately per skeleton. This
    is the faithful productionization of the POC's fixed-skeleton + arg-search:
    the two-level split is what makes the deep set-op programs discoverable."""
    frontier = [([], initial)]
    for _ in range(max_depth):
        nxt = []
        for skel, filled in frontier:
            for action in actions:
                for sargs in _structural_bindings(action):
                    if set(action.reads(sargs)) <= filled:
                        new_skel = skel + [(action, sargs)]
                        yield new_skel
                        nxt.append((new_skel, _advance(filled, Step(action, sargs))))
        frontier = nxt


def _value_params(skeleton):
    """The (step_index, Param) list of every value (non-structural) param in a
    skeleton — the axes an arg-search fills."""
    return [(i, p) for i, (action, _) in enumerate(skeleton) for p in action.params if not p.structural]


def _assemble(skeleton, values) -> Program:
    steps = []
    vi = 0
    for i, (action, sargs) in enumerate(skeleton):
        args = dict(sargs)
        for p in action.params:
            if not p.structural:
                args[p.name] = values[vi]
                vi += 1
        steps.append(Step(action, args))
    return Program(steps)


def fill_skeleton(rng, skeleton):
    """Draw a concrete `Program` from a skeleton by sampling each step's *value*
    params uniformly (structural params already fixed by the skeleton)."""
    return _assemble(skeleton, [rng.choice(p.domain) for _, p in _value_params(skeleton)])


def value_space_size(skeleton) -> int:
    """How many distinct value-arg fillings a skeleton has (1 for a param-less
    skeleton)."""
    size = 1
    for _, p in _value_params(skeleton):
        size *= len(p.domain)
    return size


def iter_fills(skeleton, rng, budget: int):
    """Up to `budget` concrete `Program`s for a skeleton: exhaustively (deduped)
    when the value space is small, else uniform random samples. This is what
    keeps arg-search efficient — a param-less skeleton costs one eval, not
    `budget` identical ones."""
    params = _value_params(skeleton)
    if value_space_size(skeleton) <= budget:
        for combo in product(*[p.domain for _, p in params]) if params else [()]:
            yield _assemble(skeleton, combo)
    else:
        for _ in range(budget):
            yield _assemble(skeleton, [rng.choice(p.domain) for _, p in params])


def sample_program(rng, actions, max_len: int) -> Program:
    """Draw one type-valid program: at each step sample uniformly among legal
    actions/args. The type constraint is what keeps the walk on the grammar
    manifold instead of the flat 0/8 space."""
    filled = frozenset({"grid"})
    steps = []
    for _ in range(rng.randint(1, max_len)):
        step = sample_step(rng, actions, filled)
        if step is None:
            break
        steps.append(step)
        filled = _advance(filled, step)
    if not steps:
        steps = [sample_step(rng, actions, frozenset({"grid"}))]
    return Program(steps)
