\# Heuristic Search Agents (AI)



Python implementations of classic AI search algorithms over a grid-based environment.



This repo focuses on \*\*correctness, optimality, and clean search logic\*\*—the kind of work that’s useful for

technical interviews and foundational AI/algorithms understanding.



\## Implemented Agents

\- \*\*BFS (Graph Search)\*\* — cycle handling with visited set

\- \*\*UCS (Dijkstra)\*\* — cost-optimal search using priority queue

\- \*\*A\\\*\*\* — heuristic + path cost for optimal planning

\- \*\*Weighted A\\\*\*\* — tunable trade-off between optimality and speed (`w`)



Each agent returns:

\- action sequence

\- total cost

\- number of expanded nodes



\## Heuristic

Uses \*\*Manhattan distance\*\* to the nearest goal (with a cap at 100), suitable for grid planning.



\## Files

\- `Algorithms.py` — agent implementations (BFS / UCS / A\* / Weighted A\*)

\- `dry.pdf` — written analysis \& answers (completeness, admissibility, complexity, comparisons)



\## Tech

\- Python 3

\- `heapdict` for priority queue behavior



\## Author

\*\*Ismael Bader\*\* (Technion CS)



