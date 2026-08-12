# Validation externe NGSIM — Synthèse finale

## Protocole

La validation externe repose sur NGSIM US-101. Les trajectoires réelles sont converties en variables compatibles avec le pipeline ML via le générateur adaptatif. Les scores ML alimentent ensuite une politique transférée puis les trois LLM.

Cette évaluation mesure la transférabilité fonctionnelle et la cohérence décisionnelle. Elle ne constitue pas une validation radio supervisée, car NGSIM ne fournit pas de vérité terrain 5G.

## Résultats communs

- Meilleur accord : DeepSeek-R1 8B (0.417).
- Modèle le plus rapide : Llama 3.2 1B (13.42 s).
- Meilleur taux de réponses exploitables : Qwen 2.5 3B (100.0%).

## Politique transférée

- MONITOR_NETWORK : 5 (25.0%)
- PREPARE_HANDOVER : 5 (25.0%)
- HANDOVER_NOW : 10 (50.0%)

## Analyse qualitative des explications

### Llama 3.2 1B

- Longueur moyenne : 45.4 mots.
- QoS : 14.3%.
- Congestion : 50.0%.
- Handover : 57.1%.
- Risque : 0.0%.
- Termes fréquents : you(11), traffic(9), select(9), decision(8), may(7), prepare_handover(6), explanation(6), congestion(5), handover(5), state(5).

### Qwen 2.5 3B

- Longueur moyenne : 42.2 mots.
- QoS : 80.0%.
- Congestion : 100.0%.
- Handover : 30.0%.
- Risque : 25.0%.
- Termes fréquents : congestion(28), load(26), reducing(20), traffic(20), high(18), quality(13), service(12), level(11), critical(11), qos(10).

### DeepSeek-R1 8B

- Longueur moyenne : 35.8 mots.
- QoS : 64.7%.
- Congestion : 94.1%.
- Handover : 82.4%.
- Risque : 41.2%.
- Termes fréquents : high(20), congestion(19), handover(15), critical(12), immediate(11), density(11), qos(9), load(9), risk(8), degradation(8).

## Comparaison interne / externe

Les différences entre les deux protocoles ne doivent pas être interprétées comme une simple baisse de performance, car le benchmark interne possède une référence différente de la validation externe. L'intérêt principal est la stabilité qualitative du compromis performance–latence et la capacité à appliquer le pipeline à une mobilité indépendante de NS-3.

## Limites

1. NGSIM ne contient pas de métriques radio 5G.
2. Certains paramètres réseau sont reconstruits par mapping adaptatif.
3. Le sous-ensemble commun est réduit par les réponses non exploitables de Llama et les timeouts de DeepSeek.
4. La politique externe est transférée depuis les scores ML, pas annotée indépendamment par des experts humains.

## Conclusion

Le pipeline est transférable à des trajectoires de mobilité réelles. DeepSeek conserve le meilleur accord décisionnel, mais avec une forte latence. Qwen est très stable dans son format de sortie mais reste biaisé vers certaines actions. Llama est rapide mais moins robuste. Ces résultats soutiennent une architecture hybride où les modèles ML structurent l'état réseau et où le LLM agit comme couche de décision explicable.
