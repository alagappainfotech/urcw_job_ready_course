"""
Module 7: Advanced Data Structures and Algorithms - Exercises
Complete these exercises to master advanced data structures and algorithms.
"""

import heapq
import time
import random
from typing import List, Any, Optional, Dict, Set, Tuple
from collections import defaultdict, deque
import math

# Exercise 1: Advanced Heap Operations
class PriorityQueue:
    """Priority queue implementation with update functionality"""
    
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0
    
    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.remove_task(task)
        
        entry = [priority, self.counter, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1
    
    def remove_task(self, task):
        """Mark an existing task as REMOVED"""
        if task in self.entry_finder:
            entry = self.entry_finder.pop(task)
            entry[-1] = self.REMOVED
    
    def pop_task(self):
        """Remove and return the lowest priority task"""
        while self.heap:
            priority, count, task = heapq.heappop(self.heap)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
    
    def is_empty(self):
        """Check if the priority queue is empty"""
        return len(self.entry_finder) == 0

# Exercise 2: Trie with Advanced Operations
class AdvancedTrie:
    """Advanced trie with additional operations"""
    
    def __init__(self):
        self.root = {}
        self.word_count = 0
    
    def insert(self, word: str, metadata: Dict = None):
        """Insert word with optional metadata"""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        
        node['$'] = {'end': True, 'metadata': metadata or {}}
        self.word_count += 1
    
    def search(self, word: str) -> bool:
        """Search for exact word match"""
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '$' in node and node['$']['end']
    
    def get_metadata(self, word: str) -> Optional[Dict]:
        """Get metadata for a word"""
        node = self.root
        for char in word:
            if char not in node:
                return None
            node = node[char]
        
        if '$' in node and node['$']['end']:
            return node['$']['metadata']
        return None
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        
        words = []
        self._dfs_collect_words(node, prefix, words)
        return words
    
    def _dfs_collect_words(self, node: Dict, current_word: str, words: List[str]):
        """DFS to collect all words from a node"""
        if '$' in node and node['$']['end']:
            words.append(current_word)
        
        for char, child_node in node.items():
            if char != '$':
                self._dfs_collect_words(child_node, current_word + char, words)
    
    def get_longest_common_prefix(self) -> str:
        """Get longest common prefix of all words"""
        if not self.root:
            return ""
        
        prefix = ""
        node = self.root
        
        while len(node) == 1 and '$' not in node:
            char = list(node.keys())[0]
            prefix += char
            node = node[char]
        
        return prefix
    
    def delete(self, word: str) -> bool:
        """Delete word from trie"""
        if not self.search(word):
            return False
        
        self._delete_recursive(self.root, word, 0)
        self.word_count -= 1
        return True
    
    def _delete_recursive(self, node: Dict, word: str, index: int) -> bool:
        """Recursively delete word from trie"""
        if index == len(word):
            if '$' in node and node['$']['end']:
                del node['$']
                return len(node) == 0
            return False
        
        char = word[index]
        if char not in node:
            return False
        
        should_delete = self._delete_recursive(node[char], word, index + 1)
        
        if should_delete:
            del node[char]
            return len(node) == 0
        
        return False

# Exercise 3: Graph Algorithms
class WeightedGraph:
    """Weighted graph with advanced algorithms"""
    
    def __init__(self, directed: bool = False):
        self.graph = defaultdict(list)
        self.directed = directed
        self.vertices = set()
    
    def add_edge(self, u: Any, v: Any, weight: float = 1):
        """Add weighted edge between vertices"""
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
    
    def dijkstra(self, start: Any, end: Any = None) -> Dict[Any, float]:
        """Dijkstra's algorithm for shortest paths"""
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        previous = {}
        pq = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == end:
                break
            
            if current_dist > distances[current]:
                continue
            
            for neighbor, weight in self.graph.get(current, []):
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances
    
    def bellman_ford(self, start: Any) -> Dict[Any, float]:
        """Bellman-Ford algorithm for shortest paths with negative weights"""
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        
        # Relax edges V-1 times
        for _ in range(len(self.vertices) - 1):
            for u in self.vertices:
                for v, weight in self.graph.get(u, []):
                    if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
        
        # Check for negative cycles
        for u in self.vertices:
            for v, weight in self.graph.get(u, []):
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    raise ValueError("Negative cycle detected")
        
        return distances
    
    def floyd_warshall(self) -> Dict[Tuple[Any, Any], float]:
        """Floyd-Warshall algorithm for all-pairs shortest paths"""
        vertices = list(self.vertices)
        n = len(vertices)
        
        # Initialize distance matrix
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        # Add edge weights
        for u in self.vertices:
            for v, weight in self.graph.get(u, []):
                u_idx = vertices.index(u)
                v_idx = vertices.index(v)
                dist[u_idx][v_idx] = weight
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Convert back to dictionary
        result = {}
        for i in range(n):
            for j in range(n):
                result[(vertices[i], vertices[j])] = dist[i][j]
        
        return result
    
    def topological_sort(self) -> List[Any]:
        """Topological sort using Kahn's algorithm"""
        if not self.directed:
            raise ValueError("Topological sort only works on directed graphs")
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        for u in self.vertices:
            for v, _ in self.graph.get(u, []):
                in_degree[v] += 1
        
        # Find vertices with no incoming edges
        queue = deque([vertex for vertex, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            
            for v, _ in self.graph.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        if len(result) != len(self.vertices):
            raise ValueError("Graph contains a cycle")
        
        return result

# Exercise 4: Advanced Sorting Algorithms
class SortingAlgorithms:
    """Collection of advanced sorting algorithms"""
    
    @staticmethod
    def counting_sort(arr: List[int], max_val: int) -> List[int]:
        """Counting sort for integers"""
        count = [0] * (max_val + 1)
        
        # Count occurrences
        for num in arr:
            count[num] += 1
        
        # Build sorted array
        result = []
        for i in range(max_val + 1):
            result.extend([i] * count[i])
        
        return result
    
    @staticmethod
    def radix_sort(arr: List[int]) -> List[int]:
        """Radix sort for integers"""
        if not arr:
            return arr
        
        max_val = max(arr)
        exp = 1
        
        while max_val // exp > 0:
            arr = SortingAlgorithms._counting_sort_by_digit(arr, exp)
            exp *= 10
        
        return arr
    
    @staticmethod
    def _counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
        """Helper function for radix sort"""
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        # Count occurrences
        for num in arr:
            index = (num // exp) % 10
            count[index] += 1
        
        # Update count array
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(n - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
        
        return output
    
    @staticmethod
    def bucket_sort(arr: List[float]) -> List[float]:
        """Bucket sort for floating point numbers"""
        if not arr:
            return arr
        
        n = len(arr)
        buckets = [[] for _ in range(n)]
        
        # Put elements in buckets
        for num in arr:
            bucket_index = int(n * num)
            if bucket_index == n:
                bucket_index = n - 1
            buckets[bucket_index].append(num)
        
        # Sort individual buckets
        for bucket in buckets:
            bucket.sort()
        
        # Concatenate buckets
        result = []
        for bucket in buckets:
            result.extend(bucket)
        
        return result

# Exercise 5: Dynamic Programming Problems
class DynamicProgramming:
    """Collection of dynamic programming solutions"""
    
    @staticmethod
    def longest_increasing_subsequence(arr: List[int]) -> int:
        """Find length of longest increasing subsequence"""
        if not arr:
            return 0
        
        n = len(arr)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if arr[j] < arr[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]
    
    @staticmethod
    def coin_change(coins: List[int], amount: int) -> int:
        """Minimum number of coins to make amount"""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    @staticmethod
    def longest_common_subsequence(text1: str, text2: str) -> int:
        """Length of longest common subsequence"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]

# Exercise 6: String Algorithms
class StringAlgorithms:
    """Collection of advanced string algorithms"""
    
    @staticmethod
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
    
    @staticmethod
    def rabin_karp_search(text: str, pattern: str, base: int = 256, mod: int = 101) -> List[int]:
        """Rabin-Karp string matching algorithm"""
        if len(pattern) > len(text):
            return []
        
        def hash_string(s: str) -> int:
            hash_value = 0
            for char in s:
                hash_value = (hash_value * base + ord(char)) % mod
            return hash_value
        
        pattern_hash = hash_string(pattern)
        text_hash = hash_string(text[:len(pattern)])
        matches = []
        
        if pattern_hash == text_hash and text[:len(pattern)] == pattern:
            matches.append(0)
        
        # Precompute base^(m-1) % mod
        h = pow(base, len(pattern) - 1, mod)
        
        for i in range(1, len(text) - len(pattern) + 1):
            text_hash = (text_hash - ord(text[i - 1]) * h) % mod
            text_hash = (text_hash * base + ord(text[i + len(pattern) - 1])) % mod
            
            if pattern_hash == text_hash and text[i:i + len(pattern)] == pattern:
                matches.append(i)
        
        return matches
    
    @staticmethod
    def manacher_algorithm(text: str) -> str:
        """Find longest palindromic substring using Manacher's algorithm"""
        if not text:
            return ""
        
        # Transform string to handle even-length palindromes
        transformed = '#' + '#'.join(text) + '#'
        n = len(transformed)
        radius = [0] * n
        center = right = 0
        max_len = 0
        max_center = 0
        
        for i in range(n):
            if i < right:
                radius[i] = min(right - i, radius[2 * center - i])
            
            # Try to expand palindrome centered at i
            try:
                while (i + radius[i] + 1 < n and 
                       i - radius[i] - 1 >= 0 and 
                       transformed[i + radius[i] + 1] == transformed[i - radius[i] - 1]):
                    radius[i] += 1
            except IndexError:
                pass
            
            # Update center and right if we found a longer palindrome
            if i + radius[i] > right:
                center = i
                right = i + radius[i]
            
            # Update maximum palindrome
            if radius[i] > max_len:
                max_len = radius[i]
                max_center = i
        
        # Extract longest palindrome
        start = (max_center - max_len) // 2
        return text[start:start + max_len]

# Exercise 7: Performance Testing Framework
class PerformanceTester:
    """Framework for testing algorithm performance"""
    
    def __init__(self):
        self.results = {}
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def benchmark_sorting_algorithms(self, data_sizes: List[int]):
        """Benchmark different sorting algorithms"""
        algorithms = {
            'bubble_sort': self._bubble_sort,
            'insertion_sort': self._insertion_sort,
            'merge_sort': self._merge_sort,
            'quick_sort': self._quick_sort,
            'heap_sort': self._heap_sort
        }
        
        results = {}
        
        for size in data_sizes:
            data = [random.randint(1, 1000) for _ in range(size)]
            results[size] = {}
            
            for name, algorithm in algorithms.items():
                test_data = data.copy()
                _, execution_time = self.time_function(algorithm, test_data)
                results[size][name] = execution_time
        
        return results
    
    def _bubble_sort(self, arr):
        """Bubble sort implementation"""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def _insertion_sort(self, arr):
        """Insertion sort implementation"""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    def _merge_sort(self, arr):
        """Merge sort implementation"""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self._merge_sort(arr[:mid])
        right = self._merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left, right):
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
    
    def _quick_sort(self, arr):
        """Quick sort implementation"""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self._quick_sort(left) + middle + self._quick_sort(right)
    
    def _heap_sort(self, arr):
        """Heap sort implementation"""
        heapq.heapify(arr)
        return [heapq.heappop(arr) for _ in range(len(arr))]

# Test Functions
def test_exercises():
    """Test all exercises"""
    print("Testing Module 7 Exercises...")
    
    # Test 1: Priority Queue
    print("\n1. Testing Priority Queue:")
    pq = PriorityQueue()
    pq.add_task("task1", 3)
    pq.add_task("task2", 1)
    pq.add_task("task3", 2)
    print(f"Popped task: {pq.pop_task()}")
    print(f"Popped task: {pq.pop_task()}")
    
    # Test 2: Advanced Trie
    print("\n2. Testing Advanced Trie:")
    trie = AdvancedTrie()
    trie.insert("hello", {"frequency": 5})
    trie.insert("world", {"frequency": 3})
    trie.insert("help", {"frequency": 2})
    print(f"Words with prefix 'hel': {trie.get_words_with_prefix('hel')}")
    print(f"Metadata for 'hello': {trie.get_metadata('hello')}")
    
    # Test 3: Graph Algorithms
    print("\n3. Testing Graph Algorithms:")
    graph = WeightedGraph(directed=True)
    graph.add_edge("A", "B", 4)
    graph.add_edge("A", "C", 2)
    graph.add_edge("B", "C", 1)
    graph.add_edge("B", "D", 5)
    graph.add_edge("C", "D", 8)
    
    distances = graph.dijkstra("A")
    print(f"Shortest distances from A: {distances}")
    
    # Test 4: Sorting Algorithms
    print("\n4. Testing Sorting Algorithms:")
    data = [64, 34, 25, 12, 22, 11, 90]
    sorted_data = SortingAlgorithms.radix_sort(data.copy())
    print(f"Original: {data}")
    print(f"Radix sorted: {sorted_data}")
    
    # Test 5: Dynamic Programming
    print("\n5. Testing Dynamic Programming:")
    arr = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_length = DynamicProgramming.longest_increasing_subsequence(arr)
    print(f"LIS length: {lis_length}")
    
    # Test 6: String Algorithms
    print("\n6. Testing String Algorithms:")
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    matches = StringAlgorithms.kmp_search(text, pattern)
    print(f"KMP matches: {matches}")
    
    # Test 7: Performance Testing
    print("\n7. Testing Performance:")
    tester = PerformanceTester()
    results = tester.benchmark_sorting_algorithms([100, 500, 1000])
    print("Performance results:")
    for size, times in results.items():
        print(f"Size {size}: {times}")
    
    print("\nAll exercises completed!")

if __name__ == "__main__":
    test_exercises()
