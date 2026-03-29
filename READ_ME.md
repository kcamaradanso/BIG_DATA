#  Spaceship Titanic : Pipeline ML Complet

> Réalisé par **Imane BOUKHARI**, **Lynda CHABOUNI**, **Léo BEN HATAT**, **Khady CAMARA DANSO**

---

##  Présentation du projet

Le dataset **Spaceship Titanic** (Kaggle) recense 12 970 passagers d'un vaisseau spatial mystérieusement téléporté dans une autre dimension. L'objectif est de prédire si chaque passager a été transporté (`Transported = True/False`) à partir de ses caractéristiques.

Ce projet couvre l'intégralité du pipeline data science : nettoyage → exploration → modélisation → ACM → clustering → export Power BI.

---

##  Structure des fichiers

```
projet/
│
├── Données/
│   ├── train.csv              # 8 693 passagers avec étiquette
│   └── test.csv               # 4 277 passagers à prédire
│
├── FINALCODE.ipynb            # Notebook principal (pipeline complet)
├── submission.csv             # Fichier de soumission Kaggle
├── spaceship_powerbi_complet.xlsx  # Tables Power BI (schéma en étoile)
└── README.md                  # Ce fichier
```

---

## ⚙️ Prérequis & Installation

### Environnement Python recommandé
```
Python 3.10+
```

### Librairies requises
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```



---

##  Description des données

| Variable | Type | Description |
|----------|------|-------------|
| `PassengerId` | string | Identifiant unique `GGGG_PP` (groupe_numéro) |
| `HomePlanet` | catégoriel | Planète d'origine (Earth, Europa, Mars) |
| `CryoSleep` | booléen | En animation suspendue pendant le voyage |
| `Cabin` | string | Cabine au format `Deck/Num/Side` |
| `Destination` | catégoriel | Planète de destination |
| `Age` | numérique | Âge du passager |
| `VIP` | booléen | Statut VIP |
| `RoomService` | numérique | Dépenses Room Service (crédits) |
| `FoodCourt` | numérique | Dépenses Food Court (crédits) |
| `ShoppingMall` | numérique | Dépenses Shopping Mall (crédits) |
| `Spa` | numérique | Dépenses Spa (crédits) |
| `VRDeck` | numérique | Dépenses VR Deck (crédits) |
| `Name` | string | Prénom et nom |
| `Transported` | **cible** | Transporté dans une autre dimension |

---

##  Pipeline : Vue d'ensemble

```
Données brutes (train + test)
        │
        ▼
1. Nettoyage & Traitement
   ├── Détection outliers (IQR) → remplacement médiane
   └── Imputation valeurs manquantes (mode / moyenne)
        │
        ▼
2. Feature Engineering
   ├── Group, GroupSize, IsAlone  ← depuis PassengerId
   ├── Deck, CabinNum, Side       ← depuis Cabin
   ├── TotalSpent, HasSpent       ← depuis dépenses
   └── FamilyName, FamilySize, HasFamily  ← depuis Name
        │
        ▼
3. Statistiques descriptives & EDA
   ├── Distributions univariées (numériques, catégorielles, binaires)
   ├── Matrice V de Cramér
   └── 8 croisements bivariés (H1–H8)
        │
        ▼
4. Modélisation (CV 5-folds stratifiée — pas de split interne)
   ├── Logistic Regression
   ├── Decision Tree (max_depth optimisé par CV)
   ├── Random Forest (RandomizedSearchCV — 30 iter.)  ← meilleur
   └── KNN (k=7)
        │
        ▼
5. ACM — Analyse des Correspondances Multiples
   ├── Classe MCA native NumPy (sans prince)
   ├── Fit sur train, transform sur train + test
   ├── Scree plot + projection individus
   └── Contributions des modalités aux axes
        │
        ▼
6. Clustering K-Means (sur axes ACM)
   ├── Méthode du coude (K=2 à 10)
   ├── K=5, n_init=20
   └── 5 clusters nommés par taux de transport
        │
        ▼
7. Export Power BI
   └── spaceship_powerbi_complet.xlsx (11 feuilles)
```

---

##  Résultats de la modélisation

**Métriques cross-validation 5-folds (métriques fiables) :**

| Modèle | AUC-ROC | Gini | F1 | Recall | Precision | Accuracy |
|--------|---------|------|----|--------|-----------|----------|
| Logistic Regression | 0.7695 | 0.5391 | 0.6884 | 0.6195 | 0.7746 | 0.7175 |
| Decision Tree | 0.7956 | 0.5912 | 0.7280 | 0.6937 | 0.7666 | 0.7391 |
| **Random Forest**  | **0.8130** | **0.6259** | **0.7306** | **0.6814** | **0.7875** | **0.7469** |
| KNN (k=7) | 0.7619 | 0.5238 | 0.6848 | 0.6530 | 0.7199 | 0.6972 |

> **Meilleur modèle : Random Forest** avec hyperparamètres optimisés (`n_estimators=300`, `max_depth=8`, `max_features=0.5`, `min_samples_split=10`).

>  Les métriques sur train complet (AUC=0.8729) sont indicatives seulement — elles reflètent une mémorisation partielle (overfitting modéré, écart de +0.06 vs CV, ce qui est acceptable).

---

##  Variables retenues pour le modèle (20 features)

| Catégorie | Variables |
|-----------|-----------|
| Profil passager | `HomePlanet`, `Destination`, `Age`, `CryoSleep`, `VIP` |
| Cabine | `Deck`, `CabinNum`, `Side` |
| Groupe | `Group`, `GroupSize`, `IsAlone` |
| Famille | `FamilySize`, `HasFamily` |
| Dépenses brutes | `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |
| Dépenses dérivées | `TotalSpent`, `HasSpent` |

---

##  ACM : Analyse des Correspondances Multiples

**Variables utilisées :** `HomePlanet`, `Destination`, `Deck`, `Side`, `CryoSleep`, `VIP`, `IsAlone`, `HasSpent`, `HasFamily`

- Implémentation **native NumPy/sklearn** — aucune dépendance externe
- Ajustement sur le **train uniquement** → transform sur train + test (pas de data leakage)
- 10 axes factoriels calculés

---

##  Clustering K-Means

5 clusters nommés par taux de transport décroissant :

| Cluster | Nom | Profil |
|---------|-----|--------|
| 🔵 | **Voyageurs Pro** | Taux transport le plus élevé |
| 🟣 | **Luxuria** | Profil dépensier, transport élevé |
| 🟢 | **Cryo Corps** | Majoritairement en CryoSleep |
| 🟡 | **Jeunes Nomades** | Transport moyen |
| 🔴 | **Colons** | Taux transport le plus faible |

---

##  Export Power BI — Schéma en Étoile

Le fichier `spaceship_powerbi_complet.xlsx` contient **11 feuilles** à importer dans Power BI :

### Tables à importer

| Feuille | Type | Lignes | Rôle |
|---------|------|--------|------|
| `F_Passagers` | **Faits** | 12 970 | Mesures numériques + clés FK |
| `D_Passager` | Dimension | 12 970 | Profil individuel |
| `D_Prediction` | Dimension | 12 970 | Résultat réel + Cluster |
| `D_Cluster` | Dimension | 5 | Segments K-Means |
| `D_Planete` | Dimension | 3 | Planètes d'origine |
| `D_Destination` | Dimension | 3 | Destinations |
| `D_Deck` | Dimension | 8 | Ponts du vaisseau |
| `D_AgeGroup` | Dimension | 5 | Tranches d'âge |
| `A_Depenses_Planete` | Agrégat | 15 | Dépenses × Planète × Service |
| `A_Depenses_Deck` | Agrégat | 40 | Dépenses × Deck × Service |
| `A_ACM_Coords` | Agrégat | 12 970 | Coordonnées ACM (10 axes) |

### Relations à créer dans Power BI

```
F_Passagers[PassengerId]       → D_Passager[PassengerId]    (N:1)
F_Passagers[PassengerId]       → D_Prediction[PassengerId]  (1:1)
F_Passagers[PlanetID]          → D_Planete[PlanetID]        (N:1)
F_Passagers[DestID]            → D_Destination[DestID]      (N:1)
F_Passagers[DeckID]            → D_Deck[DeckID]             (N:1)
F_Passagers[AgeGroupID]        → D_AgeGroup[AgeGroupID]     (N:1)
D_Prediction[ClusterID]        → D_Cluster[ClusterID]       (N:1)
A_Depenses_Planete[PlanetID]   → D_Planete[PlanetID]        (N:1)
A_Depenses_Deck[DeckID]        → D_Deck[DeckID]             (N:1)
A_ACM_Coords[PassengerId]      → D_Passager[PassengerId]    (1:1)
```

### Mesures DAX recommandées

```dax
Taux_Transport =
    DIVIDE(
        CALCULATE(COUNTROWS(F_Passagers), D_Prediction[Transported] = 1, D_Prediction[Split] = "Train"),
        CALCULATE(COUNTROWS(F_Passagers), D_Prediction[Split] = "Train")
    )

N_Transportes =
    CALCULATE(COUNTROWS(F_Passagers), D_Prediction[Transported] = 1, D_Prediction[Split] = "Train")

Moy_TotalSpent =
    AVERAGEX(F_Passagers, F_Passagers[TotalSpent])

Taux_CryoSleep =
    DIVIDE(
        CALCULATE(COUNTROWS(F_Passagers), D_Passager[CryoSleep] = "Oui"),
        COUNTROWS(F_Passagers)
    )
```

---

##  Hypothèses validées par l'EDA

| Hypothèse | Résultat |
|-----------|----------|
| H1 — CryoSleep → transport plus probable |  Confirmé (~80% transportés si CryoSleep) |
| H2 — Europa → taux transport et cryo plus élevés |  Confirmé |
| H3 — Decks B & C → fort taux CryoSleep et transport |  Confirmé |
| H4 — Europa concentre le plus de CryoSleep |  Confirmé |
| H5 — Passagers d'Europa voyagent sur des decks spécifiques |  Confirmé |
| H6 — Mars → plus souvent seul |  Confirmé |
| H7 — Deck avec fort CryoSleep → fort transport |  Confirmé |
| H8 — CryoSleep = 0 dépense (confinement cabine) |  Confirmé |

---
