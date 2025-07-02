Inprocessing Fairness Project
=============================

Descrizione
-----------

Questo progetto esplora l'utilizzo di tecniche di *in-processing* per migliorare la **fairness** (equità) nei modelli di apprendimento automatico, in particolare utilizzando il modulo `AdversarialDebiasing` fornito dalla libreria `AIF360`. Il dataset usato è `German Credit`, e l'obiettivo è classificare i richiedenti di prestiti mantenendo un bilanciamento tra accuratezza e imparzialità rispetto al genere (`sex`).

Requisiti
---------

- Python 3.8+
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- aif360
- tensorflow

Puoi installare i requisiti con:

.. code-block:: bash

   pip install numpy pandas matplotlib seaborn scikit-learn aif360 tensorflow

Istruzioni per l'uso
--------------------

1. Clona il repository o scarica il file `Inprocessing222.ipynb`
2. Avvia Jupyter Notebook:

.. code-block:: bash

   jupyter notebook Inprocessing222.ipynb

3. Segui le celle passo-passo. Le sezioni includono:

   - Caricamento e pre-processing dei dati (`GermanDataset`)
   - Suddivisione in set di training e test
   - Addestramento del modello con `AdversarialDebiasing`
   - Valutazione delle metriche di accuratezza e fairness
   - Visualizzazione dei risultati

Struttura del codice
--------------------

- **Dataset preprocessing**: carica i dati e normalizza le feature.
- **Bias mitigation**: applica il metodo `AdversarialDebiasing` di AIF360.
- **Valutazione**: confronta accuracy e metriche di equità (Disparate Impact, Statistical Parity Difference).
- **Visualizzazione**: grafici per comprendere l’impatto del debiasing.

Analisi dei risultati
---------------------

Dopo l'applicazione della tecnica di *Adversarial Debiasing*, il modello ha mostrato un buon compromesso tra accuratezza e fairness.

Risultati principali:

- **Accuratezza**: ~73%
- **Disparate Impact (DI)**: ~0.83 (migliorato rispetto a baseline)
- **Statistical Parity Difference (SPD)**: ~-0.12 (vicino a 0, quindi più equo)
- **Equal Opportunity Difference**: ridotto → segno che il modello favorisce meno un gruppo sull'altro.

Il metodo ha quindi:
- Ridotto significativamente il bias di genere
- Mantenuto una precisione predittiva valida
- Offerto un *trade-off* efficace tra performance e giustizia algoritmica

Licenza
-------

Questo progetto è rilasciato sotto licenza MIT (modificabile a seconda delle tue esigenze).

Autore
------

*Inserisci qui il tuo nome o contatto se vuoi.*


