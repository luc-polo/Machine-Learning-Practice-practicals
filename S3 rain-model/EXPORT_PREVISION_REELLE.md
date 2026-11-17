# Export nécessaire pour vérifier le CRPS de la prévision réelle

La récupération a été tentée via le compte propriétaire de la soumission.
Les routes accessibles renvoient le modèle, les scores agrégés et un replay
PNG. Elles ne renvoient ni la réponse de l'agent ni les observations complètes.
Les journaux d'exécution ne contiennent pas ces tableaux. Le PNG ne contient
aucune métadonnée numérique permettant de les récupérer.

La documentation officielle confirme la suppression de l'accès aux données
météo collectées depuis la version 4.5.0 :
[README du SDK ML-Arena](https://github.com/ml-arena/mlarena-sdk/blob/main/README.md#450--no-live-data-reads).

## Demande d'export à l'organisateur

Pour le challenge **177**, soumission **9439** (`rain-xgboost-best`),
prévision émise le **19 octobre 2026 à 11:00 UTC**, évaluée dans le run
**116987** (résultat de soumission **130596**), fournir :

- La réponse exacte enregistrée de l'agent, soit les 100 membres pour chacune
  des 45 villes et chacune des échéances +6 h et +48 h.
- Les observations de pluie exactes utilisées par le correcteur à
  **2026-10-19T17:00:00Z** et **2026-10-21T11:00:00Z**, pour les 45 villes.
- L'ordre des villes et des échéances dans ces tableaux.
- Si possible, la requête originale de l'agent, notamment son historique de
  48 heures et le masque des heures manquantes, pour contrôler aussi les entrées.

Un JSON regroupant `submission_id` (9439), `issue_time`
(`2026-10-19T11:00:00Z`), `cities` (45 noms dans l'ordre), `horizons`
(`[6, 48]`), `members` (45 × 2 × 100) et `observations` (45 × 2) suffit au calcul.

## Comparaison préparée

```sh
.venv/bin/python "S3 rain-model/compare_live_crps.py" export.json
```

Le programme applique directement la définition pair à pair du CRPS, écrit
les 90 erreurs individuelles et compare les moyennes à :

| Mesure | Valeur publiée par le challenge |
|---|---:|
| CRPS +6 h | 0.22032332000000002 |
| CRPS +48 h | 0.1774678288888889 |
| Moyenne | 0.19889557444444445 |

**Aucun recalcul sur les prévisions réelles n'a encore été effectué** : les
tableaux requis ne sont pas accessibles avec les routes et droits actuels.
Les scores ci-dessus sont ceux du serveur, pas des résultats recalculés.
La comparaison ne sera produite qu'à partir de données brutes effectivement
exportées ; ni les cartes ni une autre source météo ne sont substituées aux
observations du correcteur.
