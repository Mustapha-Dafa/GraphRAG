# GraphRAG : Approche Intelligente pour la Fiscalité Marocaine (CGI 2025)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![GraphRAG](https://img.shields.io/badge/Architecture-GraphRAG-green)

Ce projet implémente et évalue l'approche **GraphRAG** (Retrieval-Augmented Generation avec Graphes de Connaissances) appliquée au **Code Général des Impôts (CGI) marocain 2025**.

L'objectif est de dépasser les limites de la recherche par mots-clés et du RAG vectoriel classique pour fournir des réponses juridiquement justifiées, exhaustives et diversifiées aux questions fiscales complexes .

---

## Contexte et Problématique

Le CGI marocain est un corpus dense et évolutif où les règles d'assiette, de taux et de recouvrement sont interconnectées. Les méthodes classiques (Ctrl+F ou RAG vectoriel simple) peinent à capturer la logique relationnelle entre les articles de loi .

**Notre approche :** Utiliser **GraphRAG** (inspiré par Microsoft) pour modéliser le CGI sous forme de graphe, permettant de générer des réponses qui expliquent non seulement *ce que* prévoit la règle, mais aussi *pourquoi* elle s'applique, en suivant une approche "From Local to Global" .

## Méthodologie et Architecture

Nous avons comparé trois architectures pour interroger le CGI 2025 :

1.  **GraphRAG (Implémentation Microsoft) :** Construction d'un graphe de connaissances (Entités, Relations, Communautés). Utilise l'algorithme de Leiden pour créer des résumés hiérarchiques de communautés et générer des réponses globales .
2.  **RAG Classique (RC) :** Pipeline standard utilisant une base vectorielle (Baseline) .
3.  **Implémentation Simple (GR) :** Une méthode hybride personnalisée combinant graphe et vecteurs .

### Le Pipeline GraphRAG
*   **Indexation :** Extraction d'entités/relations $\rightarrow$ Création du graphe $\rightarrow$ Détection de communautés $\rightarrow$ Résumés .
*   **Interrogation :** Synthèse des résumés de communautés pour répondre à des questions thématiques globales .

## Structure du Projet

Le dépôt est organisé comme suit :

| Dossier / Fichier | Description |
| :--- | :--- |
| `GraphRAG/` & `GraphRAG2/` | Implémentation principale utilisant la bibliothèque GraphRAG de Microsoft (Meilleurs résultats). |
| `classic RAG/` | Scripts pour le pipeline RAG vectoriel de référence (Baseline). |
| `simple_implementation/` | Tentative d'implémentation hybride (Graphe + Vecteur). |
| `data/` | Contient le corpus source (CGI 2025) et les données traitées. |
| `evaluation.ipynb` | Notebook principal contenant les scripts d'évaluation comparative et les graphiques. |
| `Repenses.py` & `reponses.json` | Scripts de génération et stockage des réponses pour l'analyse. |
| `all_questions.csv` | Jeu de données des 30 questions utilisées pour l'évaluation. |

## Résultats Clés

L'évaluation a été menée sur 30 questions complexes en utilisant des critères d'Exhaustivité (Comprehensiveness) et de Diversité (Diversity).

- **Taille du Graphe :** 2 248 nœuds et 3 064 arêtes pour le CGI 2025.

### Comparaison

| Comparaison | Résultat | Analyse |
|-------------|----------|---------|
| GraphRAG vs RAG Classique | GraphRAG | GraphRAG (niveaux C0, C1, C2) surpasse significativement le RAG classique en richesse et diversité des réponses. |
| RAG Classique vs Implémentation Simple | RAG Classique | L'implémentation simple (GR) s'est avérée moins performante, soulignant l'importance de l'optimisation du pipeline Microsoft. |