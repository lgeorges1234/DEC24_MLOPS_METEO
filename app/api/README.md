Dernière modification: 12_03_2025 (Clément)

## Modification de la fonction prepare_evaluate_df():
 - Ajout de l'argument 'csv_file' et 'path_raw_data' qui va correspondre au chemin où se trouve
   le fichier .csv ainsi que le nom de se fichier à utiliser en fonction du besoin (training ou prediction).
 - Attribution dynamique du nom du fichier nettoyé (initialement "meteo.csv"): "nom_du_fichier_original" + "_cleaned.csv"

## Modification de la fonction predict() et simplifiée pour tenir compte de ce changement:
 - Appel de la fonction extract_and_prepare_df(chemin_vers_csv_daily,nom_du_csv_daily_row)
 - Appel de la fonction predict()

## Modification de la fonction train_model():

 - Nom du fichier à charger correspond au nom du fichier généré dynamiquement: "weatherAUS_training_cleaned.csv"
   au lieu de "meteo.csv"



