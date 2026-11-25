# Charte Projet — Data-Driven Energy (GreenOps & LeanOps)

## 1. Utilisateur cible et décision métier
- **Utilisateur :** Responsable énergie / gestionnaire du patrimoine bâti d'une collectivité territoriale.
- **Décision :** Identifier rapidement les dérives de consommation énergétique (surconsommation nocturne, dérèglement de chauffage/climatisation) sans attendre la facture mensuelle.
- **Valeur apportée :** Réduction des coûts énergétiques et de l'empreinte carbone grâce à une détection proactive.

## 2. Choix technique (ML vs LLM)
- **Décision :** Utilisation d'un modèle de Machine Learning classique léger (régression ou arbre de décision type LightGBM) plutôt qu'un LLM.
- **Justification :** Le problème consiste à prédire des séries temporelles numériques (kWh) à partir de données tabulaires. Le ML classique offre une meilleure précision tout en réduisant drastiquement la consommation énergétique et les coûts serveur (démarche GreenOps & LeanOps).

## 3. Périmètre du MVP (Soutenance du 7 décembre 2026)
- **Données :** Extraction et nettoyage de 3 bâtiments représentatifs d'un même campus universitaire issu du dataset public BDG2 (architecture modulaire prête pour les futures données communales).
- **Prédiction & Anomalie :** Prévision de consommation horaire/journalière et alerte en cas de dépassement d'un seuil d'écart statistique.
- **Production & Observabilité :** Déploiement conteneurisé (Docker), infrastructure provisionnée par code (IaC), suivi des performances et tableau de bord de visualisation.
