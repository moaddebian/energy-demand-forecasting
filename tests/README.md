# Tests - Energy Demand Forecasting

## Structure des Tests

```
tests/
├── conftest.py              # Configuration pytest et fixtures
├── unit/                    # Tests unitaires
│   ├── test_data_ingestion.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_api.py
├── integration/             # Tests d'intégration
│   └── test_pipeline.py
└── integration/             # Tests d'intégration
```

## Exécution des Tests

### Tous les tests
```bash
pytest
```

### Tests unitaires uniquement
```bash
pytest tests/unit/
```

### Tests d'intégration
```bash
pytest tests/integration/ -m integration
```

### Avec couverture de code
```bash
pytest --cov=src --cov-report=html
```

### Tests spécifiques
```bash
pytest tests/unit/test_api.py
pytest tests/unit/test_api.py::TestHealthEndpoint::test_health_check
```

### Mode verbeux
```bash
pytest -v
```

### Afficher les print statements
```bash
pytest -s
```

## Fixtures Disponibles

Les fixtures suivantes sont disponibles dans `conftest.py`:

- `sample_energy_data` - Données d'énergie d'exemple
- `sample_weather_data` - Données météo d'exemple
- `sample_features_df` - DataFrame de features d'exemple
- `mock_db_client` - Client de base de données mocké
- `sample_prediction_request` - Requête de prédiction d'exemple

## Marquage des Tests

Les tests peuvent être marqués avec:

- `@pytest.mark.unit` - Tests unitaires
- `@pytest.mark.integration` - Tests d'intégration
- `@pytest.mark.slow` - Tests lents

Exemple:
```python
@pytest.mark.integration
def test_end_to_end_pipeline():
    ...
```

## Configuration

La configuration pytest est dans `pytest.ini` à la racine du projet.

Les variables d'environnement de test sont définies dans `conftest.py`.

## Couverture de Code

Pour générer un rapport de couverture:

```bash
pytest --cov=src --cov-report=html
```

Le rapport HTML sera généré dans `htmlcov/index.html`.

## Notes

- Les tests utilisent des mocks pour éviter les dépendances externes
- Les tests d'intégration peuvent nécessiter une base de données de test
- Les tests API utilisent TestClient de FastAPI pour des tests rapides

