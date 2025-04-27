# Lancer les tests unitaires

## Monter l'image à partir du Dockerfile.test

commande: docker build -t weather-app-tests -f api/Dockerfile.test .

## Lancer l'application de tests

commande: docker run --rm weather-app-tests
