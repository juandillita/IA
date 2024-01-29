import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

"""
Puedes pensar que las llaves del defaultdict representan las
posiciones en el vector disperso, mientras que los valores representan
los elementos en esas posiciones.  Cualquier clave que esté ausente en
el dict significa que ese elemento en el vector disperso está ausente
(es cero).

Ten en cuenta que el tipo de llave utilizada no debería afectar al
algoritmo.  Puedes imaginar que las llaves son índices enteros (como
0, 1, 2) en el vector disperso, pero también debe funcionar igual con
llaves arbitrarias (como "red", "blue", "green").
"""
SparseVector = DefaultDict[Any, float]
Position = Tuple[int, int]


def find_alphabetically_first_word(text: str) -> str:
    """
    Dada una cadena |text|, devuelve la palabra en |text| que aparece
    primero lexicográficamente (es decir, la palabra que aparecería
    primero después de ordenarlas). Una palabra se define por una
    secuencia máxima de caracteres sin espacios en blanco. Puede que
    min() te resulte útil aquí. Si el texto de entrada es una cadena
    vacía, es aceptable devolver una cadena vacía o generar un error.
    """
    # INICIO
    if not text:
        return ""
    words = text.split()
    return min(words)
    # FIN


def euclidean_distance(loc1: Position, loc2: Position) -> float:
    """
    Regresa la distancia Euclidiana entre dos ubicaciones,
    representadas como una pareja de enteros (por ejemplo (3, 5)).
    """
    # INICIO
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    # FIN


def mutate_sentences(sentence: str) -> List[str]:
    """
    Dada una oración (secuencia de palabras), regresa una lista de
    todas las palabras "similares".
    Definimos que una oración es "similar" a la original si
      - tiene la misma cantidad de palabras, y
      - cada pareja de palabras adyacentes en la nueva oración también
        aparece en la oración original (las palabras de cada par deben
        aparecer en el mismo orden en la oración de salida que en la
        oración original).
    Notas:
      - El orden de las oraciones que produces no importa.
      - No debes producir duplicados.
      - La oración generada puede usar una palabra de la oración
        original más de una vez.
    Por ejemplo:
      - Entrada: 'the cat and the mouse'
      - Salida: ['and the cat and the', 'the cat and the mouse',
                 'the cat and the cat', 'cat and the cat and']
    """
    # INICIO
    words = sentence.split()
    if not words:
        return []
    transitions = collections.defaultdict(list)
    for i in range(len(words) - 1):
        transitions[words[i]].append(words[i + 1])

    similar_sentences = set()

    def create_sentence(current_sentence, last_word):
        if len(current_sentence.split()) == len(words):
            similar_sentences.add(current_sentence)
            return
        for next_word in transitions[last_word]:
            create_sentence(f"{current_sentence} {next_word}", next_word)

    for word in words:
        create_sentence(word, word)

    return list(similar_sentences)
    # FIN


def sparse_vector_dot_product(v1: SparseVector, v2: SparseVector) -> float:
    """
    Dados dos vectores dispersos |v1| y |v2|, cada uno representado
    como collections.defaultdict(float), regresa su producto punto.

    Puedes encontrar útil usar sum() y una comprensión de lista.  Esta
    función será útil luego para clasificadores lineales.

    Nota: Los vectores dispersos son vectores donde la mayoría de sus
    elementos son 0.
    """
    # INICIO
    return sum(v1[key] * v2[key] for key in v1 if key in v2)
    # FIN


def increment_sparse_vector(v1: SparseVector, scale: float, v2: SparseVector) -> None:
    """
    Dados dos vectores dispersos |v1| y |v2|, realiza el cálculo
    v1 += scale * v2.
    Si el valor de scale es cero, se admite modificar v1 para incluir
    cualquier llave adicional en v2, o simplemente no añadir llaves.

    Nota: Esta función debe MODIFICAR v1, pero no regresarlo.  No
    modifiques v2 en tu implementación.
    Esta función nos será útil mas adelante para clasificadores
    lineales.
    """
    # INICIO
    for key, value in v2.items():
        v1[key] += value * scale
    # FIN


def find_nonsingleton_words(text: str) -> Set[str]:
    """
    Divide la cadena |text| por espacios en blanco y regresa el
    conjunto de palabras que aparecen más de una vez.
    Puedes encontrar útil usar collections.defaultdict(int).
    """
    # INICIO
    word_counts = collections.defaultdict(int)
    for word in text.split():
        word_counts[word] += 1
        return {word for word, count in word_counts.items() if count > 1}
    # FIN
