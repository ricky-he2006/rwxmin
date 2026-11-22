import heapq

def lpt(sorted_items, k, want_workers=True):
    """sorted_items: list of (ms, test) pre-sorted descending."""
    if not want_workers:
        if k == 1:
            return sum(ms for ms, _ in sorted_items)
        h = [0] * k
        for ms, _ in sorted_items:
            heapq.heapreplace(h, h[0] + ms)
        return max(h)
    if k == 1:
        return [[t for _, t in sorted_items]], sum(ms for ms, _ in sorted_items)
    h = [[0, i] for i in range(k)]
    ws = [[] for _ in range(k)]
    for ms, t in sorted_items:
        top = h[0]
        i = top[1]
        ws[i].append(t)
        top[0] += ms
        heapq.heapreplace(h, top)
    return ws, max(load for load, _ in h)

def _refine(ws, loads):
    """Fast local search (move + swap from busiest worker) to tighten makespan."""
    k = len(ws)
    if k <= 1:
        return max(loads) if loads else 0
    for _ in range(64):
        hi = max(range(k), key=loads.__getitem__)
        cur_max = loads[hi]
        best_gain = 0
        best_move = None
        w_hi = ws[hi]
        # 1. try moving one item from hi to another worker
        for idx, (m, t) in enumerate(w_hi):
            new_hi = cur_max - m
            for j in range(k):
                if j == hi:
                    continue
                new_j = loads[j] + m
                nm = new_hi if new_hi >= new_j else new_j
                if nm < cur_max and cur_max - nm > best_gain:
                    best_gain = cur_max - nm
                    best_move = ("m", idx, j)
        # 2. try swapping one item from hi with one from another worker
        if len(w_hi) <= 64:
            for j in range(k):
                if j == hi:
                    continue
                w_j = ws[j]
                if len(w_j) > 64:
                    continue
                lj = loads[j]
                for i1, (m1, _) in enumerate(w_hi):
                    for i2, (m2, _) in enumerate(w_j):
                        if m1 <= m2:
                            continue
                        new_hi = cur_max - m1 + m2
                        new_j = lj - m2 + m1
                        nm = new_hi if new_hi >= new_j else new_j
                        if nm < cur_max and cur_max - nm > best_gain:
                            best_gain = cur_max - nm
                            best_move = ("s", i1, j, i2)
        if best_move is None:
            break
        if best_move[0] == "m":
            _, idx, j = best_move
            item = w_hi.pop(idx)
            ws[j].append(item)
            loads[hi] -= item[0]
            loads[j] += item[0]
        else:
            _, i1, j, i2 = best_move
            a, b = w_hi[i1], ws[j][i2]
            w_hi[i1], ws[j][i2] = b, a
            loads[hi] += b[0] - a[0]
            loads[j] += a[0] - b[0]
    return max(loads)


def best_schedule(tests, ms_of, want_workers=True):
    """choose worker count minimising makespan + 2000ms per worker."""
    if not tests:
        return ([], 0) if want_workers else (None, 0)
    items = sorted(((ms_of[t], t) for t in tests), reverse=True)
    best_c = 10**18
    best_k = 1
    best_ws = None
    max_k = min(8, len(items))
    total = sum(ms for ms, _ in items)
    longest = items[0][0]
    for k in range(1, max_k + 1):
        if longest + 2000 * k >= best_c:
            break
        if (total + k - 1) // k + 2000 * k >= best_c:
            continue
        if not want_workers or k == 1:
            mk = lpt(items, k, want_workers=False)
            c = mk + 2000 * k
            if c < best_c:
                best_c = c
                best_k = k
                best_ws = None
        else:
            h = [[0, i] for i in range(k)]
            ws = [[] for _ in range(k)]
            for ms, t in items:
                top = h[0]
                i = top[1]
                ws[i].append((ms, t))
                top[0] += ms
                heapq.heapreplace(h, top)
            loads = [0] * k
            for load, i in h:
                loads[i] = load
            mk = _refine(ws, loads)
            c = mk + 2000 * k
            if c < best_c:
                best_c = c
                best_k = k
                best_ws = [[t for _, t in w] for w in ws if w]
    if not want_workers:
        return None, best_c
    if best_ws is not None:
        return best_ws, best_c
    ws, _ = lpt(items, best_k, want_workers=True)
    return ws, best_c
