#!/usr/bin/python3

from typing import List, Set
import sys, itertools
from pysat.solvers import Solver
import os

########### Instance and Result

class Instance:
    def __init__(self, n: int, m: int, adj):
        self._n = n
        self._m = m
        self._adj = adj
    
    def vertex_number(self) -> int:
        return self._n

    def edge_number(self) -> int:
        return self._m
            
    def adj(self, v) -> List[int]:
        return self._adj[v]

    def set_adj(self, v, new_adj):
        self._adj[v] = new_adj
    
    def edges(self):
        for v in range(self.vertex_number()):
            for u in self.adj(v):
                if v < u:
                    yield (v, u)

    def vertex_set(self):
        return range(self.vertex_number())

    def clone(self):
        tmp = Instance(self.vertex_number(),
                        self.edge_number(),
                        [[u for u in self._adj[v]] for v in self.vertex_set()])

        return tmp
        

    
    def __repr__(self):
        return "Instance({}, {})".format(self.vertex_number(), list(self.edges()))

class Result:
    def __init__(self, depth: int, parents: List[int]):
        self._parents = parents
        self._depth = depth
        
    def depth(self):
        return self._depth

    def roots(self):
        for v in range(len(self._parents)):
            if self._parents[v] == -1:
                yield v
    
    def parent(self, i: int) -> int:
        return self._parents[i]

    def __repr__(self):
        return "Result({}, {})".format(self.depth(), self._parents)

    
def read_instance(fp) -> Instance:
    n: int = -1
    m: int = -1
    adj: List[List[int]] = None
    
    for line in fp:
        line: str = line.strip()
        if not line or line.startswith("c"):
            continue

        if line.startswith("p"):
            toks = line.split()
            n = int(toks[2])
            m = int(toks[3])
            adj = [[] for i in range(n)]
        else:
            toks = line.split()
            a = int(toks[0]) - 1
            b = int(toks[1]) - 1
            adj[a].append(b)
            adj[b].append(a)

    return Instance(n, m, adj)

def read_instance_from_args() -> Instance:
    return read_instance(sys.argv[1].split('\n'))

def write_instance(instance: Instance, fl):
    print("p tdp {} {}".format(instance.vertex_number(), instance.edge_number()), file=fl)
    for (v, u) in instance.edges():
        print("{} {}".format(v + 1, u + 1), file=fl)

def print_result(out, instance: Instance, result: Result):
    if type(result) == int:
        raise ValueError("228")
    print(result.depth())
    for i in range(instance.vertex_number()):
        print(result.parent(i) + 1)

# ###### Cover
        
# def get_cover_pulp(instance: Instance):
#     from pulp import LpProblem, LpVariable, lpSum, LpMinimize

#     print("x")
    
#     prob = LpProblem("", LpMinimize)

#     I = instance.vertex_set()
#     x = [LpVariable(str(i), cat='Binary') for i in I]
    
#     prob += lpSum(x) # objective

#     for v in I:
#         for u in instance.adj(v):
#             if v < u:
#                 prob += (x[v] + x[u] >= 1)

#     prob.solve()
#     result = [x[i].value() >= 0.99 for i in I]
#     # print("VC IS", sum(result), file=sys.stderr)
#     return result

# def get_cover(instance: Instance):
#     from ortools.linear_solver import pywraplp

#     solver = pywraplp.Solver('',
#                              pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


#     I = instance.vertex_set()
#     x = [solver.IntVar(0, solver.infinity(), str(i)) for i in I]

#     objective = solver.Minimize(sum(x))

#     for v in I:
#         for u in instance.adj(v):
#             if v < u:
#                 solver.Add(x[v] + x[u] >= 1)

#     solver.set_time_limit(20 * 1000)
#     solver.Solve()
#     result = [x[i].solution_value() >= 0.99 for i in I]
#     # print("VC IS", sum(result), file=sys.stderr)
#     return result


#### SAT-based solving

def make_solver():
    return Solver(name='Glucose4')
    
def solve_limited_with_sat(instance: Instance, mi: int) -> Result:
    if instance.vertex_number() == 0:
        return lambda: Result(0, [])
    
    solver: Solver = make_solver()

    # We are going to have N x N x length vars
    n: int = instance.vertex_number()
    length: int = mi + 1

    def flat_var(a: int, b: int, c: int) -> int:
        return 1 + ((a + b * n) * length + c)

    # basic relations:
    for v in range(n):
        for u in range(n):
            for w in range(n):
                for i in range(1, length):
                    # (v, u, i), (u, w, i) => (v, w, i)
                    # (v, w, i) or not (v, u, i) or not (u, w, i)
                    solver.add_clause([-flat_var(min(v, u), max(v, u), i),
                                       -flat_var(min(u, w), max(u, w), i),
                                       +flat_var(min(v, w), max(v, w), i)])

    # Constraint D1: P[0] is empty and P[length - 1] is full
    for v in range(n):
        for u in range(v, n):
            solver.add_clause([-flat_var(v, u, 0)])
            solver.add_clause([flat_var(v, u, length - 1)])

    # Constraint D2: P[i + 1] is refinement of P[i]
    for v in range(n):
        for u in range(v, n):
            for i in range(1, length):
                solver.add_clause([-flat_var(v, u, i - 1), flat_var(v, u, i)])


    # Constraint D3: there is at most one new vertex each time
    for v in range(n):
        for u in range(v + 1, n):
            for i in range(1, length):
                solver.add_clause([-flat_var(v, u, i), flat_var(v, v, i - 1), flat_var(u, u, i - 1)])

    # Constraint D4 each edge spanned by the tree: 
    for (v, u) in instance.edges():
        assert v < u

        for i in range(1, length):
            solver.add_clause([-flat_var(v, v, i), -flat_var(u, u, i),
                               flat_var(v, v, i - 1), flat_var(v, u, i)])

            solver.add_clause([-flat_var(v, v, i), -flat_var(u, u, i),
                               flat_var(u, u, i - 1), flat_var(v, u, i)])

    if not solver.solve():
        return None

    true_set = set(filter(lambda x: x > 0, solver.get_model()))
    def recover():
        length = mi + 1

        first_time = [-1 for i in range(n)]
        for i in range(length - 1, 0, -1):
            for v in range(n):
                if flat_var(v, v, i) in true_set:
                    first_time[v] = i

        parents = [-1 for i in range(n)]

        for (tm, v) in sorted(zip(first_time, itertools.count())):
            for u in range(n):
                if u != v and parents[u] == -1 and flat_var(min(v, u), max(v, u), tm) in true_set:
                    parents[u] = v

        # we assume here, that there was only one connected comp.
        assert parents.count(-1) == 1

        return Result(mi, parents)

    return recover

###### Kernelize

# def kernelize_add_edges(instance: Instance, p_was, VC: List[bool], td: int):
#     for v in range(instance.vertex_number()):
#         neigh = set(instance.adj(v))

#         for u in range(instance.vertex_number()):
#             if v != u and (VC[v] or VC[u]) and u not in neigh:
#                 count = 0
#                 for x in instance.adj(u):
#                     if x in neigh:
#                         count += 1

#                 if count >= td:
#                     instance.adj(v).append(u)
#                     neigh.add(u)
                    
#                     instance.adj(u).append(v)
#                     p_was[0] = True

# def kernelize_remove_vertices(instance_orig: Instance, p_was, p_recover, VC, td: int):
#     instance: Instance = instance_orig.clone()
#     removed_mark = [False for v in range(instance.vertex_number())]

#     for v in instance.vertex_set():
#         a = instance.adj(v)

#         is_good = True
#         for vert in a:
#             if len(instance.adj(vert)) <= td:
#                 is_good = False

#         if not is_good:
#             continue
        
#         for i in range(len(a)):
#             adj_set = set(instance.adj(a[i]))
#             for j in range(i):
#                 if a[j] not in adj_set:
#                     is_good = False
#                     break

#             if not is_good:
#                 break
            
#         if is_good:
#             removed_mark[v] = True

#             for u in a:
#                 instance.adj(u).remove(v)
#             instance.set_adj(v, [])

#     # # May happen only for one vertex in c.c.
#     # # Let's not get rid of it
#     # for v in instance.vertex_set():
#     #     if len(instance.adj(v)) == 0:
#     #         removed_mark[v] = True

#     if sum(removed_mark) == 0:
#         return (instance_orig, VC)

#     p_was[0] = True
#     # print("DZING", file=sys.stderr)
    
#     new_n = 0
#     new_m = 0
#     new_vs = [-1 for i in instance.vertex_set()]
#     vs_old = []
#     newVC = []
#     for v in instance.vertex_set():
#         if not removed_mark[v]:
#             new_vs[v] = new_n
#             vs_old.append(v)

#             newVC.append(VC[v])
            
#             new_n += 1
    
#     adj = [[] for i in range(new_n)]
#     new_m = 0
#     for (v, u) in instance.edges():
#         adj[new_vs[v]].append(new_vs[u])
#         adj[new_vs[u]].append(new_vs[v])
#         new_m += 1

#     oldrecover = p_recover[0]
#     def recover(result: Result) -> Result:
#         #print("was: ", Instance(new_n, new_m, adj), result, td)
        
#         newarr = [None for i in instance.vertex_set()]
#         depth = [None for i in instance.vertex_set()]
        
#         for v in instance.vertex_set():
#             if new_vs[v] != -1:
#                 tmp = result.parent(new_vs[v])
#                 if tmp == -1:
#                     newarr[v] = tmp
#                 else:
#                     newarr[v] = vs_old[tmp]

#         def calc_height(vert):
#             if depth[vert] is not None:
#                 return
            
#             if newarr[vert] == -1:
#                 depth[vert] = 1
#                 return

#             calc_height(newarr[vert])
#             depth[vert] = 1 + depth[newarr[vert]]

#         for v in instance.vertex_set():
#             if not newarr[v] is None:
#                 calc_height(v)

#         #print("depth was", depth)
#         #print("newarr", newarr)
#         #print("removed_mark", removed_mark)
        
#         for v in reversed(instance.vertex_set()):
#             if removed_mark[v]:
#                 bottom_most = (-1, -1)

#                 #print(instance_orig.adj(v))
#                 for u in instance_orig.adj(v):
#                     # print("for", v, "considering", u, depth[u])
#                     if v < u or not removed_mark[u]:
#                         bottom_most = max(bottom_most, (depth[u], u))

#                 depth[v] = bottom_most[0] + 1
#                 newarr[v] = bottom_most[1]

#         #print("to: ", instance_orig, Result(max(depth), newarr))
#         return oldrecover(Result(max(depth), newarr))

#     p_recover[0] = recover
#     # print(td, removed_mark, instance_orig, '->', Instance(new_n, new_m, adj))
#     return (Instance(new_n, new_m, adj), newVC)

# def kernelize(instance: Instance, td: int, VC):
#     p_was = [True]
#     p_recover = [lambda x: x]
#     while p_was[0]:
#         p_was[0] = False

#         kernelize_add_edges(instance, p_was, VC, td)
#         (instance, VC) = kernelize_remove_vertices(instance, p_was, p_recover, VC, td)

#     return (instance, p_recover[0])

###### Core Skeleton

def solve_limited_with_kernels(instance: Instance, mi: int, VC):
    (instance_prime, goback) = kernelize(instance.clone(), mi, VC)
    
    res = solve_limited_with_sat(instance_prime, mi)
    # res2 = solve_limited_with_sat(instance, mi)
    #
    # b1 = res == None
    # b2 = res2 == None
    # if b1 != b2:
    #    print(instance, instance_prime)
    #    raise
    
    if not res:
        return res

    return lambda: goback(res())

def solve_limited(instance: Instance, mi: int, VC):
    return solve_limited_with_sat(instance, mi)
    #return solve_limited_with_kernels(instance, mi, VC)

def transfer_to_cpp(instance: Instance, VC):
    import cffi
    ffi = cffi.FFI()
    ffi.cdef("void python_enter_point(int n, int m, int* edges, int* vc);")

    lib = ffi.dlopen("./cppsolve.so")
    lib.python_enter_point(instance.vertex_number(),
                           instance.edge_number(),
                           [instance.vertex_number() * a + b for (a, b) in instance.edges()],
                           ffi.NULL)
    
def solve(instance: Instance) -> Result:
    VC = None #VC = get_cover(instance)

    if instance.vertex_number() <= 34:
        transfer_to_cpp(instance, VC)
    
    lo: int = 0
    hi: int = 1
    recover = None
    while True:
        recover = solve_limited(instance, hi, VC)
        if recover:
            break
        
        lo = hi
        hi *= 2

    while hi - lo > 1:
        mi: int = lo + (hi - lo) // 2
        rs = solve_limited(instance, mi, VC)
        if rs:
            hi = mi
            recover = rs
        else:
            lo = mi

    print("recovering", file=sys.stderr)
    print("hi was", hi, file=sys.stderr)
    return recover()


def main():
    instance: Instance = None

    if False and len(sys.argv) > 1:
        instance = read_instance_from_args()
    else:
        instance = read_instance(sys.stdin)
    
    result: Result = solve(instance)
    print_result(sys.stdout, instance, result)

if __name__ == '__main__':
    main()
