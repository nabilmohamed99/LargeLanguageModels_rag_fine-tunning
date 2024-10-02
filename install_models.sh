#!/bin/bash

# Télécharger l'image Ollama
ollama pull mistral

# Garder le conteneur actif après le téléchargement
tail -f /dev/null
