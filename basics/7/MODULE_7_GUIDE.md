# Module 7: Advanced Data Structures and Algorithms - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master advanced data structures like heaps, tries, and graphs
- Implement efficient algorithms for sorting, searching, and graph traversal
- Understand time and space complexity analysis
- Apply data structures to solve real-world problems
- Optimize code performance using appropriate data structures
- Implement custom data structures and algorithms

## Core Concepts

### 1. Advanced Data Structures

#### Heaps and Priority Queues
```python
import heapq
from typing import List, Any, Optional

class MinHeap:
    """Min heap implementation"""
    
    def __init__(self):
        self.heap = []
    
    def push(self, item: Any, priority: float):
        """Add item to heap"""
        heapq.heappush(self.heap, (priority, item))
    
    def pop(self) -> Any:
        """Remove and return minimum item"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return heapq.heappop(self.heap)[1]
    
    def peek(self) -> Any:
        """Get minimum item without removing"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0][1]
    
    def size(self) -> int:
        """Get heap size"""
        return len(self.heap)
    
    def is_empty(self) -> bool:
        """Check if heap is empty"""
        return len(self.heap) == 0

class MaxHeap:
    """Max heap implementation using negative values"""
    
    def __init__(self):
        self.heap = []
    
    def push(self, item: Any, priority: float):
        """Add item to heap"""
        heapq.heappush(self.heap, (-priority, item))
    
    def pop(self) -> Any:
        """Remove and return maximum item"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return heapq.heappop(self.heap)[1]
    
    def peek(self) -> Any:
        """Get maximum item without removing"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0][1]
```

#### Trie (Prefix Tree)
```python
class TrieNode:
    """Trie node implementation"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.count = 0  # For counting occurrences

class Trie:
    """Trie implementation for string operations"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for word in trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        words = []
        self._dfs(node, prefix, words)
        return words
    
    def _dfs(self, node: TrieNode, current_word: str, words: List[str]):
        """Depth-first search to collect words"""
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._dfs(child_node, current_word + char, words)
    
    def delete(self, word: str) -> bool:
        """Delete word from trie"""
        if not self.search(word):
            return False
        
        node = self.root
        for char in word:
            node.children[char].count -= 1
            if node.children[char].count == 0:
                del node.children[char]
                return True
            node = node.children[char]
        
        node.is_end_of_word = False
        return True
```

#### Graph Data Structure
```python
from collections import defaultdict, deque
from typing import List, Dict, Set, Optional

class Graph:
    """Graph implementation using adjacency list"""
    
    def __init__(self, directed: bool = False):
        self.graph = defaultdict(list)
        self.directed = directed
        self.vertices = set()
    
    def add_edge(self, u: Any, v: Any, weight: float = 1):
        """Add edge between vertices u and v"""
        self.graph[u].append((v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
        
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def add_vertex(self, vertex: Any):
        """Add vertex to graph"""
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def get_neighbors(self, vertex: Any) -> List[tuple]:
        """Get neighbors of vertex"""
        return self.graph.get(vertex, [])
    
    def dfs(self, start: Any, visited: Set = None) -> List[Any]:
        """Depth-first search"""
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor, _ in self.get_neighbors(start):
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        
        return result
    
    def bfs(self, start: Any) -> List[Any]:
        """Breadth-first search"""
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                for neighbor, _ in self.get_neighbors(vertex):
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dijkstra(self, start: Any, end: Any) -> Optional[List[Any]]:
        """Dijkstra's algorithm for shortest path"""
        import heapq
        
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        previous = {}
        pq = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                return path[::-1]
            
            if current_dist > distances[current]:
                continue
            
            for neighbor, weight in self.get_neighbors(current):
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return None
```

### 2. Advanced Algorithms

#### Sorting Algorithms
```python
def merge_sort(arr: List[int]) -> List[int]:
    """Merge sort implementation"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr: List[int]) -> List[int]:
    """Quick sort implementation"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr: List[int]) -> List[int]:
    """Heap sort implementation"""
    import heapq
    
    heap = arr.copy()
    heapq.heapify(heap)
    
    result = []
    while heap:
        result.append(heapq.heappop(heap))
    
    return result
```

#### Search Algorithms
```python
def binary_search(arr: List[int], target: int) -> int:
    """Binary search implementation"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def interpolation_search(arr: List[int], target: int) -> int:
    """Interpolation search implementation"""
    left, right = 0, len(arr) - 1
    
    while left <= right and arr[left] <= target <= arr[right]:
        if left == right:
            return left if arr[left] == target else -1
        
        pos = left + int((target - arr[left]) * (right - left) / (arr[right] - arr[left]))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1
```

### 3. Dynamic Programming
```python
def fibonacci_dp(n: int) -> int:
    """Fibonacci using dynamic programming"""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

def longest_common_subsequence(text1: str, text2: str) -> int:
    """Longest common subsequence using DP"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 Knapsack problem using DP"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]
```

### 4. String Algorithms
```python
def kmp_search(text: str, pattern: str) -> List[int]:
    """KMP string matching algorithm"""
    def build_lps(pattern: str) -> List[int]:
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    lps = build_lps(pattern)
    i = j = 0
    matches = []
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

def rabin_karp_search(text: str, pattern: str) -> List[int]:
    """Rabin-Karp string matching algorithm"""
    def hash_string(s: str) -> int:
        hash_value = 0
        for char in s:
            hash_value = (hash_value * 256 + ord(char)) % 101
        return hash_value
    
    if len(pattern) > len(text):
        return []
    
    pattern_hash = hash_string(pattern)
    text_hash = hash_string(text[:len(pattern)])
    matches = []
    
    if pattern_hash == text_hash and text[:len(pattern)] == pattern:
        matches.append(0)
    
    for i in range(1, len(text) - len(pattern) + 1):
        text_hash = (text_hash - ord(text[i - 1]) * pow(256, len(pattern) - 1, 101)) % 101
        text_hash = (text_hash * 256 + ord(text[i + len(pattern) - 1])) % 101
        
        if pattern_hash == text_hash and text[i:i + len(pattern)] == pattern:
            matches.append(i)
    
    return matches
```

## Advanced Topics

### 1. Time and Space Complexity Analysis
```python
def analyze_complexity(func):
    """Decorator to analyze function complexity"""
    import time
    import tracemalloc
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory tracing
        tracemalloc.start()
        
        # Measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Function: {func.__name__}")
        print(f"Time: {end_time - start_time:.6f} seconds")
        print(f"Memory: {peak / 1024 / 1024:.2f} MB")
        
        return result
    
    return wrapper

# Example usage
@analyze_complexity
def bubble_sort(arr):
    """Bubble sort with complexity analysis"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

### 2. Custom Data Structure Implementation
```python
class CircularBuffer:
    """Circular buffer implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
    
    def enqueue(self, item: Any) -> bool:
        """Add item to buffer"""
        if self.size == self.capacity:
            return False  # Buffer full
        
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        return True
    
    def dequeue(self) -> Optional[Any]:
        """Remove item from buffer"""
        if self.size == 0:
            return None
        
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self) -> Optional[Any]:
        """Peek at next item without removing"""
        if self.size == 0:
            return None
        return self.buffer[self.head]
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size == self.capacity
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self.size == 0

class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any):
        """Put value in cache"""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
```

## Best Practices

### 1. Algorithm Selection Guidelines
```python
def choose_sorting_algorithm(data_size: int, data_type: str) -> str:
    """Choose appropriate sorting algorithm based on data characteristics"""
    if data_size < 50:
        return "insertion_sort"
    elif data_size < 1000:
        return "quick_sort"
    elif data_type == "nearly_sorted":
        return "insertion_sort"
    elif data_type == "random":
        return "merge_sort"
    else:
        return "heap_sort"

def choose_search_algorithm(data_size: int, data_type: str) -> str:
    """Choose appropriate search algorithm based on data characteristics"""
    if data_type == "sorted":
        return "binary_search"
    elif data_type == "uniform_distribution":
        return "interpolation_search"
    else:
        return "linear_search"
```

### 2. Performance Optimization
```python
def optimize_algorithm(algorithm_func, data, *args):
    """Optimize algorithm performance"""
    import cProfile
    import pstats
    from io import StringIO
    
    # Profile the algorithm
    pr = cProfile.Profile()
    pr.enable()
    result = algorithm_func(data, *args)
    pr.disable()
    
    # Get profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print("Performance Profile:")
    print(s.getvalue())
    
    return result
```

## Quick Checks

### Check 1: Heap Operations
```python
# What will this print?
heap = MinHeap()
heap.push("task1", 3)
heap.push("task2", 1)
heap.push("task3", 2)
print(heap.pop())  # What will this output?
```

### Check 2: Trie Operations
```python
# What will this return?
trie = Trie()
trie.insert("hello")
trie.insert("world")
trie.insert("help")
print(trie.get_words_with_prefix("hel"))
```

### Check 3: Graph Traversal
```python
# What will this return?
graph = Graph()
graph.add_edge("A", "B")
graph.add_edge("A", "C")
graph.add_edge("B", "D")
print(graph.dfs("A"))
```

## Lab Problems

### Lab 1: Social Network Analysis
Build a social network analysis system using graphs to find influential users and communities.

### Lab 2: Text Search Engine
Implement a text search engine using tries and advanced string matching algorithms.

### Lab 3: Task Scheduler
Create a task scheduler using priority queues and heaps to manage tasks efficiently.

### Lab 4: Pathfinding Algorithm
Implement A* pathfinding algorithm for navigation and route planning.

## AI Code Comparison
When working with AI-generated algorithm code, evaluate:
- **Correctness** - does the algorithm produce correct results?
- **Efficiency** - is the time and space complexity optimal?
- **Edge cases** - are boundary conditions handled properly?
- **Readability** - is the code clear and well-documented?
- **Scalability** - can the algorithm handle large datasets?

## Next Steps
- Learn about advanced graph algorithms and network analysis
- Master dynamic programming and optimization techniques
- Explore parallel and distributed algorithms
- Study machine learning algorithms and data structures
- Understand algorithm design patterns and trade-offs
