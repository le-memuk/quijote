"""
preparar_datos.py — Prepara datos de texto y código Python para entrenar
========================================================================
Descarga código Python de GitHub y limpia los datos para entrenamiento.

Uso:
  python preparar_datos.py texto     # prepara solo libros de datos/
  python preparar_datos.py codigo    # descarga código Python de GitHub
  python preparar_datos.py todo      # ambos
  python preparar_datos.py mios      # indexa tus propios .py de datos_codigo/
"""

import os
import sys
import re
import json
import time
import random
import urllib.request
import urllib.error
from pathlib import Path


# ══════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════

CARPETA_TEXTO  = Path("datos")
CARPETA_CODIGO = Path("datos_codigo")
CARPETA_FINAL  = Path("datos_entrenamiento")   # donde va todo junto limpio

# Repositorios Python pequeños y educativos de GitHub (sin API key)
# Son proyectos públicos con código Python limpio y bien comentado
REPOS_PYTHON = [
    # Algoritmos y estructuras de datos — muy buenos para aprender patrones
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/bubble_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/quick_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/merge_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/binary_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/linear_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/linked_list/singly_linked_list.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/stacks/stack.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/queue/queue_on_list.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/breadth_first_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/depth_first_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/fibonacci.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/prime_check.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/factorial.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/palindrome.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/anagram.py",
    # Matemáticas y ciencia
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/matrix_multiplication.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/newton_raphson.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/gaussian.py",
    # Machine learning básico
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/linear_regression.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/k_nearest_neighbours.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/decision_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/naive_bayes.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/gradient_descent.py",

    # Más algoritmos de ordenamiento
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/insertion_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/selection_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/heap_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/counting_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/radix_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/shell_sort.py",

    # Estructuras de datos avanzadas
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/binary_search_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/avl_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/heap/heap.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/linked_list/doubly_linked_list.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/trie/trie.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/hashing/hash_table.py",

    # Grafos
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/dijkstra.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/floyd_warshall.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/topological_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/bellman_ford.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/minimum_spanning_tree_kruskal.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/prim.py",

    # Matemáticas adicionales
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/euler_method.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/prime_sieve_of_eratosthenes.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/greatest_common_divisor.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/least_common_multiple.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/abs.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/average.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/ceil.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/floor.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/modular_exponential.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/binary_exp_mod.py",

    # Strings / procesamiento de texto
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/levenshtein_distance.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/rabin_karp.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/knuth_morris_pratt.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/boyer_moore_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/longest_common_substring.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/word_occurrence.py",

    # Programación dinámica
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/fibonacci.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/knapsack.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/longest_common_subsequence.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/coin_change.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/matrix_chain_order.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/edit_distance.py",

    # Criptografía básica
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/caesar_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/vigenere_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/base64_encoding.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/rot13.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/morse_code.py",

    # Redes neuronales básicas (muy útil para el modelo)
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/neural_network/perceptron.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/neural_network/back_propagation_neural_network.py",

    # geekcomputers — scripts útiles del mundo real
    "https://raw.githubusercontent.com/geekcomputers/Python/master/Colors/colors.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/check_website_is_up/check_website.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/Word_Count/wordcount.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/calculator/simple_calculator.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/Fibonacci/fibonacci.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/Palindrome/palindrome.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/Temperature/temp_conversion.py",
    "https://raw.githubusercontent.com/geekcomputers/Python/master/BMI/bmi.py",

    # TheAlgorithms — más algoritmos grandes y detallados
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/tim_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/bitonic_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/cycle_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/gnome_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/strand_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/pancake_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/pigeonhole_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/double_sort.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/jump_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/interpolation_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/exponential_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/fibonacci_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/ternary_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/tabu_search.py",

    # Grafos adicionales
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/a_star.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/articulation_points.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/strongly_connected_components.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/tarjans_scc.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/check_bipartite_graph_bfs.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/eulerian_path_and_circuit.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/kahns_algorithm_long.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/greedy_best_first.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/multi_heuristic_astar.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/page_rank.py",

    # Árboles adicionales
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/red_black_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/segment_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/fenwick_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/treap.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/lazy_segment_tree.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/number_of_possible_binary_trees.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/queue/circular_queue.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/queue/priority_queue_using_list.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/linked_list/circular_linked_list.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/linked_list/skip_list.py",

    # Matemáticas adicionales
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/extended_euclidean_algorithm.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/chinese_remainder_theorem.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/collatz_sequence.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/hardy_ramanujam.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/lucas_series.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/mobius_function.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/numerical_analysis/bisection.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/numerical_analysis/secant_method.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/numerical_analysis/simpson_rule.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/numerical_analysis/trapezoid_rule.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/numerical_analysis/runge_kutta.py",

    # Machine learning adicional
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/logistic_regression.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/random_forest_classifier.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/support_vector_machines.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/k_means_clust.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/dbscan.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/polynomial_regression.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/lasso_regression.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/machine_learning/ridge_regression.py",

    # Redes neuronales adicionales
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/neural_network/convolution_neural_network.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/neural_network/gan.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/neural_network/input_data.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/neural_network/two_hidden_layers_neural_network.py",

    # Strings adicionales
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/naive_string_search.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/z_function.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/aho_corasick.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/manacher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/min_cost_string_conversion.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/capitalize.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/check_pangram.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/frequency_finder.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/is_pangram.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/jaro_winkler.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/lower.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/upper.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/snake_case_to_camel_pascal_case.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/split.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/strings/strip.py",

    # Programación dinámica adicional
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/climbing_stairs.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/house_robber.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/max_subarray.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/minimum_partition.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/rod_cutting.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/word_break.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/subset_generation.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/bitmask.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/abbreviation.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/all_construct.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/combination_sum_iv.py",

    # Criptografía adicional
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/hill_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/playfair_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/rail_fence_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/transposition_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/xor_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/rsa_cipher.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/diffie_hellman.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/elgamal_key_generator.py",

    # Backtracking
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/n_queens.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/sudoku.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/knight_tour.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/permutations.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/subsets.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/sum_of_subsets.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/coloring.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/hamiltonian_cycle.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/all_combinations.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/all_permutations.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/generate_parentheses.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/backtracking/minimax.py",

    # Divide y vencerás
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/divide_and_conquer/closest_pair_of_points.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/divide_and_conquer/inversions.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/divide_and_conquer/max_subarray.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/divide_and_conquer/strassen_matrix_multiplication.py",

    # Bit manipulation
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/single_bit_manipulation_operations.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/count_1s_brian_kernighan_method.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/binary_and_operator.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/binary_or_operator.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/binary_xor_operator.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/reverse_bits.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/bit_manipulation/swap_all_odd_and_even_bits.py",

    # Problemas de arrays
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/matrix/matrix_operation.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/matrix/rotate_matrix.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/matrix/spiral_print.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/matrix/count_islands_in_matrix.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/matrix/pascal_triangle.py",

    # Diseño de software
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/two_sum.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/anagram_checker.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/tower_of_hanoi.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/password_generator.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/magicdiamondpattern.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/finding_primes.py",
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/other/detecting_english_programmatically.py",
]

# Snippets de código Python escritos a mano para que la IA aprenda
# patrones de pregunta-respuesta en código
SNIPPETS_PROPIOS = [
    # Formato: (descripción, código)
    ("# Función para calcular el factorial de un número\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # 120\n"),
    ("# Clase para representar una pila (stack)\nclass Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        if not self.is_empty():\n            return self.items.pop()\n\n    def is_empty(self):\n        return len(self.items) == 0\n\n    def peek(self):\n        return self.items[-1] if not self.is_empty() else None\n"),
    ("# Leer un archivo de texto línea por línea\ndef leer_archivo(ruta):\n    try:\n        with open(ruta, 'r', encoding='utf-8') as f:\n            return f.readlines()\n    except FileNotFoundError:\n        print(f'Archivo no encontrado: {ruta}')\n        return []\n"),
    ("# Ordenar una lista de diccionarios por un campo\npersonas = [\n    {'nombre': 'Ana', 'edad': 25},\n    {'nombre': 'Luis', 'edad': 30},\n    {'nombre': 'Eva', 'edad': 22},\n]\nordenado = sorted(personas, key=lambda x: x['edad'])\nprint(ordenado)\n"),
    ("# Contar palabras en un texto\ndef contar_palabras(texto):\n    palabras = texto.lower().split()\n    conteo = {}\n    for palabra in palabras:\n        conteo[palabra] = conteo.get(palabra, 0) + 1\n    return conteo\n"),
    ("# Conexión simple a una base de datos SQLite\nimport sqlite3\n\ndef crear_tabla():\n    conn = sqlite3.connect('mi_base.db')\n    cursor = conn.cursor()\n    cursor.execute('''\n        CREATE TABLE IF NOT EXISTS usuarios (\n            id INTEGER PRIMARY KEY,\n            nombre TEXT NOT NULL,\n            email TEXT UNIQUE\n        )\n    ''')\n    conn.commit()\n    conn.close()\n"),
    ("# Descargar una imagen de internet\nimport urllib.request\n\ndef descargar_imagen(url, ruta_destino):\n    try:\n        urllib.request.urlretrieve(url, ruta_destino)\n        print(f'Imagen guardada en {ruta_destino}')\n    except Exception as e:\n        print(f'Error: {e}')\n"),
    ("# Generador de números de Fibonacci\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n\nfor num in fibonacci(10):\n    print(num)\n"),
    ("# Decorador para medir tiempo de ejecución\nimport time\n\ndef medir_tiempo(func):\n    def wrapper(*args, **kwargs):\n        inicio = time.time()\n        resultado = func(*args, **kwargs)\n        fin = time.time()\n        print(f'{func.__name__} tardó {fin - inicio:.4f} segundos')\n        return resultado\n    return wrapper\n\n@medir_tiempo\ndef operacion_lenta():\n    time.sleep(1)\n\noperacion_lenta()\n"),
    ("# Clase para trabajar con matrices\nclass Matriz:\n    def __init__(self, filas, cols):\n        self.datos = [[0] * cols for _ in range(filas)]\n        self.filas = filas\n        self.cols  = cols\n\n    def set(self, i, j, val):\n        self.datos[i][j] = val\n\n    def get(self, i, j):\n        return self.datos[i][j]\n\n    def __str__(self):\n        return '\\n'.join([str(fila) for fila in self.datos])\n"),

    # JSON
    ("# Leer y escribir archivos JSON\nimport json\n\ndef leer_json(ruta):\n    with open(ruta, 'r', encoding='utf-8') as f:\n        return json.load(f)\n\ndef escribir_json(datos, ruta):\n    with open(ruta, 'w', encoding='utf-8') as f:\n        json.dump(datos, f, ensure_ascii=False, indent=2)\n\ndatos = {'nombre': 'Quijote', 'capitulos': 52}\nescribir_json(datos, 'datos.json')\nprint(leer_json('datos.json'))\n"),

    # List comprehensions
    ("# List comprehensions en Python\ncuadrados = [x**2 for x in range(20) if x % 2 == 0]\npares = list(filter(lambda x: x % 2 == 0, range(20)))\ncuadrados_dict = {x: x**2 for x in range(10)}\nconjunto = {x % 5 for x in range(20)}\nprint(cuadrados)\nprint(pares)\nprint(cuadrados_dict)\nprint(conjunto)\n"),

    # Excepciones
    ("# Manejo de excepciones en Python\ndef dividir(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        print('Error: division entre cero')\n        return None\n    except TypeError:\n        print('Error: argumentos deben ser numeros')\n        return None\n    finally:\n        print('Operacion completada')\n\nprint(dividir(10, 2))\nprint(dividir(10, 0))\n"),

    # Generadores
    ("# Generadores con yield\ndef fibonacci_gen():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\ndef primeros_n(gen, n):\n    return [next(gen) for _ in range(n)]\n\nprint(primeros_n(fibonacci_gen(), 10))\n"),

    # Herencia
    ("# Herencia y polimorfismo\nclass Figura:\n    def area(self):\n        raise NotImplementedError\n\n    def perimetro(self):\n        raise NotImplementedError\n\nclass Rectangulo(Figura):\n    def __init__(self, ancho, alto):\n        self.ancho = ancho\n        self.alto  = alto\n\n    def area(self):\n        return self.ancho * self.alto\n\n    def perimetro(self):\n        return 2 * (self.ancho + self.alto)\n\nclass Circulo(Figura):\n    import math\n    def __init__(self, radio):\n        self.radio = radio\n\n    def area(self):\n        return 3.14159 * self.radio ** 2\n\n    def perimetro(self):\n        return 2 * 3.14159 * self.radio\n\nfiguras = [Rectangulo(4, 5), Circulo(3)]\nfor f in figuras:\n    print(f'Area: {f.area():.2f} | Perimetro: {f.perimetro():.2f}')\n"),

    # Regex
    ("# Expresiones regulares\nimport re\n\ntexto = 'Contacto: usuario@ejemplo.com o llama al 555-1234'\n\nemail = re.search(r'[\\w.]+@[\\w.]+\\.[a-z]{2,}', texto)\ntelefono = re.search(r'\\d{3}-\\d{4}', texto)\n\nif email:\n    print(f'Email encontrado: {email.group()}')\nif telefono:\n    print(f'Telefono: {telefono.group()}')\n\n# Reemplazar todos los numeros\nsin_numeros = re.sub(r'\\d+', '#', texto)\nprint(sin_numeros)\n"),

    # Dataclasses
    ("# Dataclasses modernas\nfrom dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass Estudiante:\n    nombre: str\n    edad:   int\n    notas:  List[float] = field(default_factory=list)\n\n    def promedio(self):\n        return sum(self.notas) / len(self.notas) if self.notas else 0.0\n\n    def aprobo(self):\n        return self.promedio() >= 6.0\n\ne = Estudiante('Ana', 20, [8.5, 9.0, 7.5])\nprint(e)\nprint(f'Promedio: {e.promedio():.1f} - Aprobo: {e.aprobo()}')\n"),

    # Threading
    ("# Hilos con threading\nimport threading\nimport time\n\nresultados = []\nlock = threading.Lock()\n\ndef calcular(n):\n    time.sleep(0.1)\n    with lock:\n        resultados.append(n * n)\n\nhilos = [threading.Thread(target=calcular, args=(i,)) for i in range(10)]\nfor h in hilos:\n    h.start()\nfor h in hilos:\n    h.join()\n\nprint(sorted(resultados))\n"),

    # Argparse
    ("# Argumentos de linea de comandos\nimport argparse\n\nparser = argparse.ArgumentParser(description='Procesador de archivos')\nparser.add_argument('archivo', help='Archivo de entrada')\nparser.add_argument('--salida', '-o', default='resultado.txt')\nparser.add_argument('--verbose', '-v', action='store_true')\nparser.add_argument('--numero', '-n', type=int, default=10)\n\nargs = parser.parse_args()\nif args.verbose:\n    print(f'Procesando {args.archivo} -> {args.salida}')\n"),

    # SQLite
    ("# Base de datos SQLite\nimport sqlite3\n\ndef crear_db(nombre):\n    conn = sqlite3.connect(nombre)\n    c = conn.cursor()\n    c.execute('''\n        CREATE TABLE IF NOT EXISTS productos (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            nombre TEXT NOT NULL,\n            precio REAL,\n            stock  INTEGER DEFAULT 0\n        )\n    ''')\n    conn.commit()\n    return conn\n\ndef insertar(conn, nombre, precio, stock):\n    conn.cursor().execute(\n        'INSERT INTO productos (nombre, precio, stock) VALUES (?, ?, ?)',\n        (nombre, precio, stock)\n    )\n    conn.commit()\n\ndef buscar(conn, nombre):\n    c = conn.cursor()\n    c.execute('SELECT * FROM productos WHERE nombre LIKE ?', (f'%{nombre}%',))\n    return c.fetchall()\n"),
]


# ══════════════════════════════════════════════════════════
#  LIMPIEZA DE CÓDIGO
# ══════════════════════════════════════════════════════════

def limpiar_codigo_python(codigo: str) -> str:
    """
    Limpia código Python para entrenamiento:
    - Elimina líneas en blanco excesivas
    - Normaliza indentación
    - Elimina caracteres extraños
    - Mantiene comentarios (son útiles para aprender)
    """
    lineas = codigo.split('\n')
    limpias = []
    blancos_consecutivos = 0

    for linea in lineas:
        # Máximo 2 líneas en blanco consecutivas
        if linea.strip() == '':
            blancos_consecutivos += 1
            if blancos_consecutivos <= 2:
                limpias.append('')
        else:
            blancos_consecutivos = 0
            # Eliminar caracteres no ASCII problemáticos
            linea = linea.encode('ascii', errors='ignore').decode('ascii')
            if linea.strip():
                limpias.append(linea)

    return '\n'.join(limpias)


def limpiar_texto(texto: str) -> str:
    """Limpia texto para entrenamiento."""
    # Normalizar espacios y saltos de línea
    texto = re.sub(r'\r\n', '\n', texto)
    texto = re.sub(r'\n{4,}', '\n\n\n', texto)
    texto = re.sub(r' {4,}', ' ', texto)
    # Eliminar caracteres de control excepto newline y tab
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)
    return texto.strip()


# ══════════════════════════════════════════════════════════
#  DESCARGA DE CÓDIGO
# ══════════════════════════════════════════════════════════

def descargar_codigo_github():
    """Descarga archivos Python de GitHub."""
    CARPETA_CODIGO.mkdir(exist_ok=True)
    descargados = 0
    fallidos    = 0

    print(f"📥 Descargando {len(REPOS_PYTHON)} archivos Python de GitHub...")

    for url in REPOS_PYTHON:
        nombre = url.split('/')[-1]
        destino = CARPETA_CODIGO / f"github_{nombre}"

        if destino.exists():
            print(f"  ⏭️  {nombre} ya existe")
            continue

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                codigo = resp.read().decode('utf-8', errors='ignore')

            codigo_limpio = limpiar_codigo_python(codigo)
            if len(codigo_limpio) > 100:   # ignorar archivos muy pequeños
                destino.write_text(codigo_limpio, encoding='utf-8')
                print(f"  ✅ {nombre} ({len(codigo_limpio)} chars)")
                descargados += 1
            else:
                print(f"  ⚠️  {nombre} muy pequeño, ignorado")

            time.sleep(0.3)   # respetar rate limit de GitHub

        except Exception as e:
            print(f"  ❌ {nombre}: {e}")
            fallidos += 1

    print(f"\n📊 Descargados: {descargados} | Fallidos: {fallidos}")
    return descargados


def guardar_snippets_propios():
    """Guarda los snippets escritos a mano."""
    CARPETA_CODIGO.mkdir(exist_ok=True)
    ruta = CARPETA_CODIGO / "snippets_propios.py"
    contenido = "\n\n# " + "─" * 50 + "\n\n".join(SNIPPETS_PROPIOS)
    ruta.write_text(contenido, encoding='utf-8')
    print(f"✅ {len(SNIPPETS_PROPIOS)} snippets propios guardados")


# ══════════════════════════════════════════════════════════
#  PREPARACIÓN FINAL
# ══════════════════════════════════════════════════════════

def preparar_datos_finales():
    """
    Une texto y código en la carpeta final de entrenamiento.
    Añade marcadores especiales para que el modelo aprenda
    a distinguir texto de código:

      <texto> ... </texto>
      <codigo> ... </codigo>
    """
    CARPETA_FINAL.mkdir(exist_ok=True)

    total_chars = 0

    # ── Textos ─────────────────────────────────────────
    textos_out = []
    if CARPETA_TEXTO.exists():
        for txt in sorted(CARPETA_TEXTO.glob("*.txt")):
            try:
                contenido = txt.read_text(encoding='utf-8', errors='ignore')
                contenido = limpiar_texto(contenido)
                if contenido:
                    textos_out.append(f"<texto>\n{contenido}\n</texto>")
                    total_chars += len(contenido)
                    print(f"  📄 {txt.name}: {len(contenido):,} chars")
            except Exception as e:
                print(f"  ❌ {txt.name}: {e}")

    if textos_out:
        salida = "\n\n".join(textos_out)
        (CARPETA_FINAL / "textos.txt").write_text(salida, encoding='utf-8')
        print(f"✅ Textos: {len(textos_out)} archivos → datos_entrenamiento/textos.txt")

    # ── Código Python ──────────────────────────────────
    codigos_out = []
    if CARPETA_CODIGO.exists():
        for py in sorted(CARPETA_CODIGO.glob("*.py")):
            try:
                contenido = py.read_text(encoding='utf-8', errors='ignore')
                contenido = limpiar_codigo_python(contenido)
                if contenido:
                    codigos_out.append(f"<codigo>\n{contenido}\n</codigo>")
                    total_chars += len(contenido)
                    print(f"  🐍 {py.name}: {len(contenido):,} chars")
            except Exception as e:
                print(f"  ❌ {py.name}: {e}")

    if codigos_out:
        salida = "\n\n".join(codigos_out)
        (CARPETA_FINAL / "codigo.txt").write_text(salida, encoding='utf-8')
        print(f"✅ Código: {len(codigos_out)} archivos → datos_entrenamiento/codigo.txt")

    total_mb = total_chars / 1_000_000
    print(f"\n📊 Total: {total_mb:.2f} MB de datos de entrenamiento")

    if total_mb < 5:
        print("⚠️  Menos de 5MB — agrega más libros o código para mejores resultados")
    elif total_mb < 20:
        print("👍 Cantidad aceptable — el modelo aprenderá razonablemente bien")
    else:
        print("🚀 Excelente cantidad de datos")

    return total_mb


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    comando = sys.argv[1].lower() if len(sys.argv) > 1 else "todo"

    print("=" * 55)
    print("  🗡️  Preparador de datos — IA Quijotesca 300M")
    print("=" * 55)

    if comando in ("codigo", "todo"):
        print("\n🐍 Preparando datos de código Python...")
        guardar_snippets_propios()
        descargar_codigo_github()

    if comando in ("mios",):
        print("\n📁 Indexando tus propios .py de datos_codigo/...")
        if not CARPETA_CODIGO.exists() or not list(CARPETA_CODIGO.glob("*.py")):
            print("⚠️  No hay archivos .py en datos_codigo/")
            print("   Copia tus archivos .py ahí y vuelve a ejecutar")
            return

    if comando in ("texto", "todo", "mios"):
        print("\n📚 Preparando datos finales...")
        preparar_datos_finales()

    print("\n✅ Listo. Ahora entrena con:")
    print("   python chat_quijote.py")
    print("\n   (asegúrate de que CARPETA en chat_quijote.py")
    print("    apunte a 'datos_entrenamiento')")


if __name__ == "__main__":
    main()