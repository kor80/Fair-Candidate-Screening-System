1. Data Cleaning (Selection)
=============================

In questa fase iniziale, il dataset grezzo viene sottoposto a un processo di selezione e pulizia al fine di ottenere una base coerente, priva di rumore e adatta alla fase di classificazione.

Obiettivo
---------

- Preparare un dataset utilizzabile per la fase di **classificazione**
- Rimuovere dati inconsistenti, duplicati, o con troppi valori mancanti
- Applicare filtri logici o statistici per escludere record non rilevanti

Operazioni eseguite
-------------------

- Rimozione dei record con più del **30% di valori mancanti**
- Eliminazione di **feature irrilevanti** (es. ID, timestamp non usati, commenti testuali liberi)
- Conversione di colonne categoriche in formato leggibile o codificato
- Uniformazione dei tipi di dato (es. trasformazione di numeri float stringati in numerici)

Esito
-----

Il dataset risultante è stato ridotto da **X** feature iniziali a **Y** feature selezionate, con un totale di **N** istanze (record), pronte per essere analizzate nella fase successiva.

Questo dataset è stato validato per assenza di duplicati, coerenza dei tipi di dato e sufficiente densità di informazione per procedere alla classificazione.
