import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# CLASIFICACIÓN BINARIA
############################################################

############################################################
# Extracción de características


def extractWordFeatures(x: str) -> FeatureVector:
    # INICIO
    features = {}
    for word in x.split():
        features[word] = features.get(word, 0) + 1
    return features
    # FIN


############################################################
# Descenso de gradiente estocástico

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    """
    Dado |trainExamples| y |validationExamples| que son listas de
    pares (x,y), un |featureExtractor| para aplicar a x, y el número
    de épocas para entrenar |numEpochs|, el tamaño de paso |eta|,
    regresa el vector de pesos (como un vector disperso) que se haya
    aprendido.

    Debes implementar descenso de gradiente estocástico.
    Notas:
    - ¡Solo utiliza trainExamples para entrenamiento!
    - Debes llamar evaluatePredictor() sobre trainExamples y
      validationExamples para ver cómo vas conforme aprendes después
      de cada época.
    - El predictor debe producir +1 si la respuesta es precisamente 0.
    """
    weights = {}  # característica => peso

    # INICIO
    def dotProduct(d1, d2):
        return sum(d1.get(f, 0) * v for f, v in d2.items())

    def increment(d1, scale, d2):
        for feature, value in d2.items():
            d1[feature] = d1.get(feature, 0) + value * scale

    for epoch in range(numEpochs):
        for x, y in trainExamples:
            features = featureExtractor(x)
            prediction = dotProduct(weights, features)
            error = prediction - y  # The gradient for squared loss
            increment(weights, -eta * error, features)

        trainError = evaluatePredictor(trainExamples,
                                       lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        validationError = evaluatePredictor(validationExamples,
                                            lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print(f"Epoch {epoch}: Train Error = {trainError}, Validation Error = {validationError}")
    # FIN
    return weights


############################################################
# Generar casos de prueba


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Regresa un conjunto de ejemplos (phi(x), y) aleatoriamente
    pero clasificados correctamente por |weights|.
    """
    random.seed(42)

    # Regresa un único ejemplo (phi(x), y).
    # phi(x) debe ser un diccionario cuyas llaves son un subconjunto
    # de las llaves en los pesos y los valores pueden ser cualquier cosa
    # con una respuesta para el vector de pesos dado.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # INICIO
        raise Exception("Not implemented yet")
        # FIN
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Características de caracteres


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Regresa una función que toma una cadena |x| y regresa un vector
    de características disperso que consiste de todos los n-gramas
    de |x| sin espacios y asociado al conteo de su n-grama.
    Por ejemplo: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...}
    Puedes suponer que n >= 1.
    """
    def extract(x: str) -> Dict[str, int]:
        # INICIO
        raise Exception("Not implemented yet")
        # FIN

    return extract


############################################################
# Para el problema 3.5.


def testValuesOfN(n: int):
    """
    Usa este código para probar diferentes valores de n para
    extractCharacterFeatures, este código es únicamente para
    pruebas. Tu respuesta completa al problema 3.5 debe estar en
    tarea1.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# K-medias
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    """
    Realiza agrupamiento con K-medias sobre |examples|, donde cada
    ejemplo es un vector de características disperso.

    @params
    - examples: una lista de ejemplos, cada ejemplo es un diccionario
      de cadena --> flotante representando un vector disperso.
    - K: número de grupos deseados. Supon que 0 < K <= |examples|.
    - maxEpochs: maxima cantidad de épocas para correr (deber terminar
      antes si el algoritmo converge).
    @return una lista de tamaño K con los centroides de los grupos,
    una lista de asignaciones tales que si examples[i] pertenece a
    centers[j], entonces assignments[i] = j, y la pérdida de
    reconstrucción final.
    """
    # INICIO
    raise Exception("Not implemented yet")
    # FIN
