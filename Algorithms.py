import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict

from collections import deque
from typing import List, Tuple, Optional


class Node:
    __slots__ = ("state", "parent", "action", "step_cost")

    def __init__(
        self,
        state: int,
        parent: Optional["Node"] = None,
        action: Optional[int] = None,
        step_cost: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.step_cost = step_cost


def return_sol(node: Node, expanded: int):
    actions, cost = [], 0.0
    while node.parent is not None:
        actions.append(node.action)
        cost += node.step_cost
        node = node.parent
    actions.reverse()
    return actions, cost, expanded


class BFSGAgent:
    """
    Breadth-First Search agent rewritten so the code is new,
    but outward behaviour (actions, total_cost, expanded) is unchanged.
    """

    def search(self, env: "HaifaEnv") -> Tuple[List[int], float, int]:
        env.reset()
        start = env.get_initial_state()

        if env.is_final_state(start):
            return [], 0.0, 1

        frontier: deque[Node] = deque([Node(start)])
        visited: set[int] = {start}
        expanded = 0

        while frontier:
            current = frontier.popleft()
            expanded += 1

            env.set_state(current.state)
            for act, succ_info in env.succ(current.state).items():
                if not succ_info:                 # empty tuple/list
                    continue

                succ_state = succ_info[0]         # first field = state
                if succ_state is None:            # <--- guard added
                    continue

                succ_cost = succ_info[1]          # second field = step cost
                if succ_state in visited:
                    continue

                child = Node(succ_state, current, act, succ_cost)

                if env.is_final_state(succ_state):
                    return return_sol(child, expanded)

                frontier.append(child)
                visited.add(succ_state)

        # no solution found
        return [], -1.0, -1
import heapdict
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Minimal search-tree node (matches the BFS version you already adopted)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Uniform-Cost Search (Dijkstra)
# ---------------------------------------------------------------------------
class UCSAgent:
    """
    Uniform-Cost Search that behaves just like your original implementation
    (same tie-breaking, same expanded-node count) but with the slim Node.
    """

    # ------------ helper: path reconstruction ---------------------------
    @staticmethod
    def _reconstruct(goal: Node, expanded: int) -> Tuple[List[int], float, int]:
        actions, cost = [], 0.0
        n = goal
        while n.parent is not None:
            actions.insert(0, n.action)
            cost += n.step_cost
            n = n.parent
        return actions, cost, expanded

    # ------------ main search -------------------------------------------
    def search(self, env: "HaifaEnv") -> Tuple[List[int], float, int]:
        env.reset()
        start = env.get_initial_state()

        if env.is_final_state(start):
            return [], 0.0, 0

        frontier = heapdict.heapdict()
        root = Node(start)
        frontier[root] = (0.0, start)          # (priority, tie-break key)

        best_cost = {start: 0.0}               # cheapest g(n) per state
        closed = set()                         # tracks *expanded* states
        expanded = 0

        while frontier:
            current, (g_current, _) = frontier.popitem()
            s = current.state

            # Same stale-entry guard as original code
            if g_current > best_cost[s]:
                continue

            # Goal test *before* counting expansion (matches original)
            if env.is_final_state(s):
                return self._reconstruct(current, expanded)

            # Mark node as expanded
            closed.add(s)
            expanded += 1
            env.set_state(s)

            # Generate successors
            for act, succ_info in env.succ(s).items():
                if not succ_info:
                    continue

                s_next = succ_info[0]
                if s_next is None:
                    continue

                step = succ_info[1]
                g_next = g_current + step

                # If we found a cheaper route to s_next
                if g_next < best_cost.get(s_next, float("inf")):
                    best_cost[s_next] = g_next
                    child = Node(s_next, current, act, step_cost=step)
                    frontier[child] = (g_next, s_next)   # same tie-breaker

        # No path found
        return [], -1.0, expanded

class WeightedAStarAgent:
    """
    Weighted A*    (f = (1-w)·g + w·h ,   where 0 ≤ w ≤ 1)
    Returns (actions, total_cost, expanded_nodes).
    """

    # ---------- heuristic (Manhattan distance, clipped at 100) ----------
    @staticmethod
    def _heuristic(env: "HaifaEnv", state: int) -> float:
        goal_rc = [env.to_row_col(g) for g in env.get_goal_states()]
        r, c = env.to_row_col(state)
        dist = min(abs(r - gr) + abs(c - gc) for gr, gc in goal_rc)
        return min(dist, 100)

    # ---------- path reconstruction ------------------------------------
    @staticmethod
    def _reconstruct(goal: Node, expanded: int) -> Tuple[List[int], float, int]:
        actions, cost = [], 0.0
        n = goal
        while n.parent is not None:
            actions.insert(0, n.action)
            cost += n.step_cost
            n = n.parent
        return actions, cost, expanded

    # ---------- main search --------------------------------------------
    def search(self, env: "HaifaEnv", h_weight: float) -> Tuple[List[int], float, int]:
        """
        Parameters
        ----------
        env       : HaifaEnv   – environment instance
        h_weight  : float      – w in  f = (1-w)·g + w·h   (0 ≤ w ≤ 1)

        Returns
        -------
        actions   : List[int]
        total_cost: float
        expanded  : int
        """
        env.reset()
        start = env.get_initial_state()

        if env.is_final_state(start):
            return [], 0.0, 0

        # frontier:  key = Node,   value = (f, tie_break)
        frontier = heapdict.heapdict()

        g_best = {start: 0.0}                         # best g(s) seen
        h0 = self._heuristic(env, start)
        f0 = (1.0 - h_weight) * 0.0 + h_weight * h0
        root = Node(start)
        frontier[root] = (f0, start)

        closed = set()                                # expanded states
        expanded = 0

        while frontier:
            current, (f_curr, _) = frontier.popitem()
            s = current.state
            g_curr = g_best[s]                        # cheapest path so far

            # Stale-path guard
            if (1.0 - h_weight) * g_curr + h_weight * self._heuristic(env, s) < f_curr:
                continue

            # Goal test
            if env.is_final_state(s):
                return self._reconstruct(current, expanded)

            # Expand
            closed.add(s)
            expanded += 1
            env.set_state(s)

            for act, succ_info in env.succ(s).items():
                if not succ_info:
                    continue
                s_next = succ_info[0]
                if s_next is None:
                    continue

                step = succ_info[1]
                g_next = g_curr + step
                if g_next >= g_best.get(s_next, float("inf")):
                    continue                          # not an improvement

                # Better path found
                g_best[s_next] = g_next
                h_next = self._heuristic(env, s_next)
                f_next = (1.0 - h_weight) * g_next + h_weight * h_next
                child = Node(s_next, current, act, step_cost=step)
                frontier[child] = (f_next, s_next)    # tie-break by state id

        # No solution
        return [], -1.0, expanded
class AStarAgent():
    def __init__(self) -> None:
        self.env = None

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        wagent = WeightedAStarAgent()
        # As seen in the tutorial
        return wagent.search(env, 0.5)
