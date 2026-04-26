import pygame
import collections
import heapq
import sys

CELL_SIZE    = 64
MAP_SIZE     = 10
PANEL_WIDTH  = 220
WINDOW_W     = MAP_SIZE * CELL_SIZE + PANEL_WIDTH
WINDOW_H     = MAP_SIZE * CELL_SIZE
FPS          = 60
STEP_DELAY   = 60  

BG_DARK      = (18, 18, 24)    
GRID_LINE    = (40, 40, 50)    
WHITE        = (240, 240, 245) 
STATIC_WALL  = (45, 45, 60)    
EXPLORED     = (100, 150, 255, 100) 
FRONTIER     = (255, 200, 100) 
PATH_COLOR   = (0, 255, 150)   
START_COLOR  = (255, 255, 0)   
TARGET_COLOR = (255, 0, 255)   
CURRENT_CLR  = (255, 255, 255) 
PANEL_BG     = (25, 25, 35)    
TEXT_COLOR   = (200, 200, 220)

EMPTY   =  0
STATIC  = -1

class AIPathfinder:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.map         = [[EMPTY] * MAP_SIZE for _ in range(MAP_SIZE)]
        self.start       = (7, 7)
        self.target      = (8, 1)

        self.directions = [
            (-1,  0),   #  Up
            ( 0,  1),   #  Right
            ( 1,  0),   #  Bottom
            ( 1,  1),   #  Bottom-Right (Diagonal)
            ( 0, -1),   #  Left
            (-1, -1),   #  Top-Left (Diagonal)
        ]

        for row in range(2, 8):
            self.map[row][5] = STATIC

    def is_walkable(self, cell):
        row, col = cell
        return (0 <= row < self.map_size and 0 <= col < self.map_size and
                self.map[row][col] != STATIC)

    def trace_back_path(self, came_from, end_node):
        route = []
        current = end_node
        while current is not None:
            route.append(current)
            current = came_from.get(current)
        return route[::-1]

    def bfs_gen(self):
        to_visit  = collections.deque([self.start])
        came_from = {self.start: None}
        visited   = []
        while to_visit:
            current = to_visit.popleft()
            if current == self.target:
                yield {"current": current, "visited": visited, "frontier": list(to_visit), 
                       "path": self.trace_back_path(came_from, self.target), "done": True}
                return
            if current not in visited:
                visited.append(current)
                for dr, dc in self.directions:
                    neighbor = (current[0] + dr, current[1] + dc)
                    if self.is_walkable(neighbor) and neighbor not in came_from:
                        came_from[neighbor] = current
                        to_visit.append(neighbor)
            yield {"current": current, "visited": list(visited), "frontier": list(to_visit), "path": None, "done": False}
        yield {"done": True, "failed": True}

    def dfs_gen(self, depth_limit=None):
        to_visit  = [(self.start, 0)]
        came_from = {self.start: None}
        visited   = []
        while to_visit:
            current, current_depth = to_visit.pop()
            if current == self.target:
                yield {"current": current, "visited": visited, "frontier": [n for n,_ in to_visit], 
                       "path": self.trace_back_path(came_from, self.target), "done": True}
                return
            if current not in visited:
                if depth_limit is None or current_depth < depth_limit:
                    visited.append(current)
                    for dr, dc in reversed(self.directions):
                        neighbor = (current[0] + dr, current[1] + dc)
                        if self.is_walkable(neighbor) and neighbor not in came_from:
                            came_from[neighbor] = current
                            to_visit.append((neighbor, current_depth + 1))
            yield {"current": current, "visited": list(visited), "frontier": [n for n,_ in to_visit], "path": None, "done": False}
        yield {"done": True, "failed": True}

    def ucs_gen(self):
        pq = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        visited = []
        while pq:
            current_cost, current = heapq.heappop(pq)
            if current == self.target:
                yield {"current": current, "visited": visited, "frontier": [n for _,n in pq], 
                       "path": self.trace_back_path(came_from, self.target), "done": True}
                return
            if current not in visited:
                visited.append(current)
                for dr, dc in self.directions:
                    neighbor = (current[0] + dr, current[1] + dc)
                    new_cost = current_cost + 1
                    if self.is_walkable(neighbor):
                        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                            cost_so_far[neighbor] = new_cost
                            came_from[neighbor] = current
                            heapq.heappush(pq, (new_cost, neighbor))
            yield {"current": current, "visited": list(visited), "frontier": [n for _,n in pq], "path": None, "done": False}
        yield {"done": True, "failed": True}

    def iddfs_gen(self):
        for limit in range(1, self.map_size * self.map_size):
            yield {"current": self.start, "visited": [], "frontier": [], "path": None, "done": False}
            gen = self.dfs_gen(depth_limit=limit)
            for state in gen:
                if state.get("done"):
                    if not state.get("failed"):
                        yield state
                        return
                else:
                    yield state
        yield {"done": True, "failed": True}

    def bidirectional_gen(self):
        f_q = collections.deque([self.start]); f_came = {self.start: None}; f_vis = set()
        b_q = collections.deque([self.target]); b_came = {self.target: None}; b_vis = set()
        def get_bidir_path(meet):
            p1 = self.trace_back_path(f_came, meet)
            p2 = []
            curr = meet
            while curr:
                curr = b_came.get(curr)
                if curr: p2.append(curr)
            return p1 + p2
        while f_q and b_q:
            curr_f = f_q.popleft()
            f_vis.add(curr_f)
            for dr, dc in self.directions:
                n = (curr_f[0] + dr, curr_f[1] + dc)
                if self.is_walkable(n) and n not in f_came:
                    f_came[n] = curr_f
                    f_q.append(n)
                    if n in b_came:
                        yield {"current": n, "visited": list(f_vis|b_vis), "frontier": list(f_q)+list(b_q), 
                               "path": get_bidir_path(n), "done": True}; return
            curr_b = b_q.popleft()
            b_vis.add(curr_b)
            for dr, dc in self.directions:
                n = (curr_b[0] + dr, curr_b[1] + dc)
                if self.is_walkable(n) and n not in b_came:
                    b_came[n] = curr_b
                    b_q.append(n)
                    if n in f_came:
                        yield {"current": n, "visited": list(f_vis|b_vis), "frontier": list(f_q)+list(b_q), 
                               "path": get_bidir_path(n), "done": True}; return
            yield {"current": curr_f, "visited": list(f_vis|b_vis), "frontier": list(f_q)+list(b_q), "path": None, "done": False}
        yield {"done": True, "failed": True}

class PygameApp:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Traversal Showing")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.pf      = AIPathfinder()
        self.state   = None
        self.gen     = None
        self.running_algo = None
        self.paused  = False
        self.step_timer = 0

    def start_algo(self, key):
        self.pf = AIPathfinder()
        self.running_algo = key
        self.paused = False
        self.state = None
        if key == "bfs":     self.gen = self.pf.bfs_gen()
        elif key == "dfs":   self.gen = self.pf.dfs_gen()
        elif key == "ucs":   self.gen = self.pf.ucs_gen()
        elif key == "dls":   self.gen = self.pf.dfs_gen(depth_limit=15)
        elif key == "iddfs": self.gen = self.pf.iddfs_gen()
        elif key == "bidir": self.gen = self.pf.bidirectional_gen()

    def draw_grid(self):
        visited = set(self.state["visited"]) if self.state else set()
        frontier = set(self.state["frontier"]) if self.state else set()
        path = set(self.state["path"] or []) if self.state else set()
        current = self.state["current"] if self.state else None
        for r in range(MAP_SIZE):
            for c in range(MAP_SIZE):
                x, y = c * CELL_SIZE, r * CELL_SIZE
                color = WHITE
                val = self.pf.map[r][c]
                if val == STATIC: color = STATIC_WALL
                elif (r,c) == self.pf.start: color = START_COLOR
                elif (r,c) == self.pf.target: color = TARGET_COLOR
                elif (r,c) in path: color = PATH_COLOR
                elif (r,c) == current: color = CURRENT_CLR
                elif (r,c) in visited: color = EXPLORED
                elif (r,c) in frontier: color = FRONTIER
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, GRID_LINE, (x, y, CELL_SIZE, CELL_SIZE), 1)

    def draw_panel(self):
        panel_rect = pygame.Rect(MAP_SIZE * CELL_SIZE, 0, PANEL_WIDTH, WINDOW_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        panel_x = MAP_SIZE * CELL_SIZE + 20
        info_lines = [
            "  TRAVERSAL SHOWING",
            "  -----------------",
            "  ROLL NUMBERS:",
            "  24F-0733",
            "  24F-0639",
            "  -----------------",
            "  [B] BFS",
            "  [D] DFS",
            "  [U] UCS",
            "  [L] DLS (15)",
            "  [I] IDDFS",
            "  [S] BIDIR",
            "  -----------------",
            "  [P] PAUSE",
            "  [R] RESET",
            "  -----------------",
            f"  ACTIVE: {str(self.running_algo or 'NONE').upper()}"
        ]
        for i, line in enumerate(info_lines):
            lbl = self.font_sm.render(line, True, TEXT_COLOR)
            self.screen.blit(lbl, (panel_x, 40 + i * 22))

    def tick_algo(self):
        if self.gen is None or self.paused: return
        if self.state and self.state.get("done"): return
        now = pygame.time.get_ticks()
        if now - self.step_timer < STEP_DELAY: return
        self.step_timer = now
        try:
            self.state = next(self.gen)
        except StopIteration: pass

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b: self.start_algo("bfs")
                    elif event.key == pygame.K_d: self.start_algo("dfs")
                    elif event.key == pygame.K_u: self.start_algo("ucs")
                    elif event.key == pygame.K_l: self.start_algo("dls")
                    elif event.key == pygame.K_i: self.start_algo("iddfs")
                    elif event.key == pygame.K_s: self.start_algo("bidir")
                    elif event.key == pygame.K_p: self.paused = not self.paused
                    elif event.key == pygame.K_r: 
                        self.pf = AIPathfinder(); self.gen = None; self.state = None; self.running_algo = None
            self.tick_algo()
            self.screen.fill(BG_DARK)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()

if __name__ == "__main__":
    PygameApp().run()