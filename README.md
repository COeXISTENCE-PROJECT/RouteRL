**State** (not used right now, just defined):

|       | Origin 1              | Origin 2 | Origin 3 | ... | Origin n
|-------|-----------------------|----------|----------|-----|---------
| Destination 1 | Path 1/ Path 2/ Path 3| Paths | Paths | Paths |Paths
| Destination 2 | Paths             | Paths | Paths | Paths |Paths
| Destination 3 | Paths             | Paths | Paths | Paths |Paths
| ...           | Paths             | Paths | Paths | Paths |Paths
| Destination n | Paths             | Paths | Paths | Paths |Paths


**Observation:**

The observations are found by checking the agent's origin and destination pairs. 
Humans with the same origin-destination pair are searched for.
Actions are taken for these agents, and calculations are made regarding how many of them choose each path.
In the end, the observations will be represented by an array where cells denote the number of different paths, and values are determined by the number of vehicles that chose each path.


| Path 1             | Path 2 | Path 3 | ...| Path n
|--------------------|--------|--------|-----|------
|                    |        |        |

