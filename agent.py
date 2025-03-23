# agent.py for IDA*
import pygame

class Agent(pygame.sprite.Sprite):
    def __init__(self, environment, grid_size):
        super().__init__()
        self.image = pygame.Surface((grid_size, grid_size))
        self.image.fill((0, 0, 255))  # Agent color is blue
        self.rect = self.image.get_rect()
        self.grid_size = grid_size
        self.environment = environment
        self.position = [0, 0]  # Starting at the top-left corner of the grid
        self.rect.topleft = (0, 0)
        self.task_completed = 0
        self.completed_tasks = []
        self.path = []  # List of positions to follow
        self.moving = False  # Flag to indicate if the agent is moving
        self.path_cost = 0  # Total path cost
        self.task_costs = {}  # Store costs for completed tasks
        
    def move(self):
        """Move the agent along the path."""
        if self.path:
            next_position = self.path.pop(0)
            self.position = list(next_position)
            self.rect.topleft = (self.position[0] * self.grid_size, self.position[1] * self.grid_size)
            self.check_task_completion()
        else:
            self.moving = False  # Stop moving when path is exhausted
            
    def check_task_completion(self):
        """Check if the agent has reached a task location."""
        position_tuple = tuple(self.position)
        if position_tuple in self.environment.task_locations:
            task_number = self.environment.task_locations.pop(position_tuple)
            self.task_completed += 1
            self.completed_tasks.append(task_number)
            # Record the cost for this task
            task_cost = self.path_cost
            if self.task_costs:
                task_cost -= sum(self.task_costs.values())
            self.task_costs[task_number] = task_cost

    def find_nearest_task(self):
        """Find the nearest task location using Manhattan distance."""
        if not self.environment.task_locations:
            return None
            
        # Find task with minimum Manhattan distance from current position
        current_pos = tuple(self.position)
        min_distance = float('inf')
        nearest_task = None
        
        for task_pos in self.environment.task_locations.keys():
            distance = self.manhattan_distance(current_pos, task_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_task = task_pos
                
        return nearest_task

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def find_path_to_task(self, target):
        """Find a path to the target position using IDA* algorithm."""
        start = tuple(self.position)
        goal = target
        
        # Initialize path and visited set
        path = [start]
        visited = set([start])
        
        # Initial threshold is the heuristic from start to goal
        threshold = self.manhattan_distance(start, goal)
        
        found_path = None
        
        # Main IDA* loop
        while found_path is None:
            # Distance to exceed before we stop current iteration
            min_exceed = float('inf')
            
            # Stack for depth-first search
            stack = [(path, visited, 0)]
            
            while stack and found_path is None:
                current_path, current_visited, cost = stack.pop()
                current = current_path[-1]
                
                # Calculate f = g + h
                f = cost + self.manhattan_distance(current, goal)
                
                # If f exceeds threshold, update min_exceed
                if f > threshold:
                    min_exceed = min(min_exceed, f)
                    continue
                    
                # Goal check
                if current == goal:
                    found_path = current_path
                    break
                    
                # Get neighbors in reverse order for DFS
                neighbors = self.get_neighbors(current[0], current[1])
                neighbors.reverse()  # To explore in the same order as BFS when using stack
                
                for neighbor in neighbors:
                    if neighbor not in current_visited:
                        # Add neighbor to path and visited
                        new_path = current_path + [neighbor]
                        new_visited = current_visited.copy()
                        new_visited.add(neighbor)
                        
                        # Add to stack
                        stack.append((new_path, new_visited, cost + 1))
            
            # If no path found, increase threshold to min_exceed
            if found_path is None:
                if min_exceed == float('inf'):
                    # No path exists
                    self.moving = False
                    return
                threshold = min_exceed
            
        # Path found, update agent
        self.path = found_path[1:]  # Exclude the current position
        self.moving = True
        self.path_cost += len(self.path)  # Update total path cost
            
    def get_neighbors(self, x, y):
        """Get walkable neighboring positions."""
        neighbors = []
        directions = [("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))]
        
        for _, (dx, dy) in directions:
            nx, ny = x + dx, y + dy
            if self.environment.is_within_bounds(nx, ny) and not self.environment.is_barrier(nx, ny):
                neighbors.append((nx, ny))
                
        return neighbors
    