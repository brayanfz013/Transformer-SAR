from features.preprocess import PrepareImageUtils
from lib.class_load import LoadFiles
import yaml
from pathlib import Path
import argparse
from typing import Any


def buscar_valores(lista_a, lista_b):
    """
    Encuentra los valores de "lista_a" que están en "lista_b" y los que no.

    Parámetros:
      lista_a (list): La lista con los valores a buscar.
      lista_b (list): La lista donde se busca.

    Retorno:
      tuple: Dos listas, la primera con los valores encontrados y la segunda con los que no.
    """
    # Convertir las listas a conjuntos para búsquedas más rápidas
    conjunto_a = set(lista_a)
    conjunto_b = set(lista_b)

    # Encontrar la intersección (valores en ambas listas)
    interseccion = conjunto_a & conjunto_b

    # Encontrar la diferencia (valores solo en "lista_b")
    diferencia = conjunto_b - conjunto_a

    # Convertir los conjuntos a listas para el retorno
    lista_contiene = list(interseccion)
    lista_nocontiene = list(diferencia)

    return lista_contiene, lista_nocontiene


def load_parameters() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml-path",
        "-y",
        type=Path,
        default=Path("data/config/config.yaml"),
    )
    args = parser.parse_args()
    return args


def main():
    load_args = load_parameters()
    config_path: Path = load_args.yaml_path

    yaml_config: dict[str, dict[str, Path | str | float]] = yaml.safe_load(
        config_path.open()
    )
    DATASET_FOLDER = Path(yaml_config["preprocessing"]["image-path-raw"])
    NOISY_FOLDER = DATASET_FOLDER.joinpath("Noisy")
    GTRUTH = DATASET_FOLDER.joinpath("GTruth")
    SAVE_DATA = DATASET_FOLDER.joinpath(yaml_config["preprocessing"]["output"])
    SAVED_DATA_GTRUTH = SAVE_DATA.joinpath("GTruth")
    SAVED_DATA_NOISY = SAVE_DATA.joinpath("Noisy")

    handler_files = LoadFiles()
    handler_image = PrepareImageUtils()

    # Buscar imagen en el directorio
    print("Buscar imagen en el directorio")
    folder_noisy = handler_files.search_load_files_extencion(
        path_search=NOISY_FOLDER.as_posix(), ext=["tiff"]
    )["tiff"][1]
    folder_gtruth = handler_files.search_load_files_extencion(
        path_search=GTRUTH.as_posix(), ext=["tiff"]
    )["tiff"][1]

    # Crear las imagenes aumentadas agregandole un prefijo
    print("Crear las imagenes aumentadas agregandole un prefijo")
    # TODO: Pasar a Multiprocessing
    for noisy, gtruth in zip(folder_noisy, folder_gtruth):
        handler_image.create_augmented_data(
            noisy, aumented_data=SAVED_DATA_NOISY, preffix_name="_a"
        )
        handler_image.create_augmented_data(
            gtruth, aumented_data=SAVED_DATA_GTRUTH, preffix_name="_a"
        )

    # Cambio de extencion para tener todo en el mismo formato
    print("Cambio de extencion para tener todo en el mismo formato")
    for noisy, gtruth in zip(folder_noisy, folder_gtruth):
        handler_image.change_extencion(noisy, "png", SAVED_DATA_NOISY)
        handler_image.change_extencion(gtruth, "png", SAVED_DATA_GTRUTH)

    # Buscando imagenes aumentadas y originales en formato PNG
    print("Buscando imagenes aumentadas y originales en formato PNG ")
    folder_noisy = handler_files.search_load_files_extencion(
        path_search=SAVED_DATA_NOISY.as_posix(), ext=["png"]
    )["png"][1]
    folder_gtruth = handler_files.search_load_files_extencion(
        path_search=SAVED_DATA_GTRUTH.as_posix(), ext=["png"]
    )["png"][1]

    # Separando aleatoriamente train y test
    print("Separando aleatoriamente train y test ")
    test_original, train_original = handler_image.create_split_train_test_data(
        folder_noisy, 0.8
    )

    print("Buscando Complementos de las imagenes")
    train_gt, test_gt = buscar_valores(
        [Path(image).name for image in list(train_original)],
        [Path(image).name for image in folder_gtruth],
    )

    train_gt = [SAVED_DATA_GTRUTH / image for image in train_gt]
    test_gt = [SAVED_DATA_GTRUTH / image for image in test_gt]

    parameters_move = [
        (train_original, SAVED_DATA_NOISY / "train"),
        (test_original, SAVED_DATA_NOISY / "test"),
        (train_gt, SAVED_DATA_GTRUTH / "train"),
        (test_gt, SAVED_DATA_GTRUTH / "test"),
    ]

    print("Moviendo imagenes a las carpetas destino de train y test")
    for images, path in parameters_move:
        handler_image.move_images(images, path)


if __name__ == "__main__":
    main()
