# -*- coding: utf-8 -*-
# =============================================================================
__author__ = "Alejandro Amar y Brayan Felipe Zapata "
__copyright__ = "Copyright 2024"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Brayan Felipe Zapata"
__email__ = "abnolecture@gmail.com"
__status__ = "Production"
__doc__ = "Clase para aumentar imagenes"
# =============================================================================
import os
import shutil
from pathlib import Path
from random import random, sample

import numpy as np
import PIL
from PIL import Image


class PrepareImageUtils:
	def __init__(self):
		self._transforms_values = {
			"horizontal_flip": round(random(), 2),
			"vertical_flip": round(random(), 2),
			"rotation": round(random() * 100),
		}

	def transform_image(self, image: Image) -> Image:
		image_ = (
			image.transpose(Image.FLIP_TOP_BOTTOM)
			if self._transforms_values["horizontal_flip"] > 0.5
			else image
		)
		image_ = (
			image_.transpose(Image.FLIP_LEFT_RIGHT)
			if self._transforms_values["vertical_flip"] > 0.5
			else image
		)
		image_ = image_.rotate(
			self._transforms_values["rotation"], PIL.Image.NEAREST, expand=1
		)
		image_ = image_.crop(
			box=(
				image_.size[0] / 2 - image.size[0] / 2,
				image_.size[1] / 2 - image.size[1] / 2,
				image_.size[0] / 2 + image.size[0] / 2,
				image_.size[1] / 2 + image.size[1] / 2,
			)
		)

		return image_

	def create_augmented_data(
		self, images_to_aumented: Path, aumented_data: Path, preffix_name: str
	) -> None:
		"""
		This function takes an image file, applies transformations to it, and saves the augmented image with
		a specified prefix name in a specified directory.

		Args:
		  images_to_aumented (Path): `images_to_aumented` is the path to the original image that you want to
		augment.
		  aumented_data (Path): `aumented_data` is a Path object representing the directory where the
		augmented data will be saved.
		  preffix_name (str): The `preffix_name` parameter is a string that will be used as a prefix for the
		saved augmented images. It will be added to the stem of the original image file name to create a new
		file name for the augmented image.
		"""
		save_name: str = Path(images_to_aumented).stem + preffix_name
		img_noisy = Image.open(images_to_aumented)
		save_image = self.transform_image(img_noisy)
		save_path: str = (
			aumented_data.joinpath(save_name).with_suffix(".png").as_posix()
		)
		save_image.save(save_path)
		# cv2.imwrite(save_path,np.asarray(save_image))

	def create_split_train_test_data(
		self, list_annotation: list, percentage_split: float
	):
		"""create_split_train_test_data Funcion para generar las listas de separacion
		de archivos, generado una lista con un item para el train y otro para el test

		Args:
		    dict_annotation (dict): Diccionario completo con las anotaciones
		    percentage_split (int): Porcentaje de separacion de la informacion

		Returns:
		    _type_: lista con los valores que se requiren separar
		"""

		# Listado image
		nombre_imagenes = list(list_annotation)
		# Calculo del total de imagenes
		total_images = len(nombre_imagenes)
		images = np.array(nombre_imagenes)
		# Seleccion de la cantidad de imagenes para las pruebas
		test_index = np.array(
			sample(range(total_images), int(total_images * percentage_split))
		)
		test_image = images[test_index]
		# Seleccion de la cantidad de imagenes para el entranamiento
		train_index = range(0, total_images)
		train_index = np.delete(train_index, test_index, None)
		train_images = images[train_index]
		split_data = [train_images, test_image]

		return split_data

	def create_folder_data(self, save_split_data: str):
		"""create_folder_data Genera las carpetas para guardar el entrenamiento

		Args:
		    save_split_data (str): ruta donde se guardan las carpetas train y test

		    folder (list): Retorna la lista de las carpetas creadas
		"""
		names = ["train", "test"]
		folder_list = [
			os.path.join(save_split_data, names[i]) for i in range(len(names))
		]
		for i in folder_list:
			if not os.path.isdir(i):
				try:
					os.mkdir(i)
					os.chmod(i, 0o777)
					print("[INFO] Directorio:" + i + " Creado!")

				except OSError as error:
					print(f"[INFO] Data folder was created in:\n{i}{error}")
		return folder_list

	def change_extencion(
		self, image: str, extension: str, save_path: str | Path = None
	) -> bool:
		"""
		This Python function changes the extension of an image file if it exists and is able to be loaded
		and saved successfully.

		Args:
		    image (str): The `image` parameter in the `change_extension` method is a string that represents
		the file path of the image file that you want to change the extension of.
		    extension (str): The `extension` parameter in the `change_extension` method is a string that
		represents the new extension you want to change the image file to. It should be provided without the
		leading dot (e.g., "jpg" instead of ".jpg"). The method will handle adding the dot if it's missing

		Returns:
		    a boolean value - True if the image extension was successfully changed, and False if there was an
		error or if the extension was already the same as the desired one.
		"""
		if not os.path.isfile(image):
			print(f"File not found in path: {image}")
			return False

		image_name = Path(image)
		if save_path is None:
			save_path = image_name.parent
		else:
			save_path = Path(save_path)

		if not extension.startswith("."):
			extension: str = f".{extension}"

		if image_name.suffix.lower() == extension.lower():
			print(f"Extesion alredy is {extension} ")
			return False

		try:
			image_ = Image.open(image)
		except IOError as e:
			print(f"Error in load image {e}")
			return False

		try:
			image_.save(save_path.joinpath(image_name.name).with_suffix(extension))
		except IOError as e:
			print(f"Erro to save image {e}")
			return False
		return True

	def move_file(self, ruta_origen: str | Path, ruta_destino: str | Path) -> bool:
		# Verificar que el archivo original exista
		if not os.path.isfile(ruta_origen):
			print(f"El archivo '{ruta_origen}' no existe.")
			return False

		# Verificar que el directorio destino exista
		directorio_destino = os.path.dirname(ruta_destino)
		if not os.path.isdir(directorio_destino):
			try:
				os.makedirs(directorio_destino)
			except OSError as e:
				print(f"Error al crear el directorio destino: {e}")
				return False

		# Mover el archivo
		try:
			os.rename(ruta_origen, ruta_destino)
		except OSError as e:
			print(f"Error al mover el archivo: {e}")
			return False

		print(f"El archivo se movió correctamente a '{ruta_destino}'.")
		return True

	def move_images(self, src: list[Path], folder_name: Path):
		folder_name.mkdir(parents=True, exist_ok=True)
		[shutil.move(image, folder_name) for image in src]

	def buscar_valores(self, lista_a, lista_b):
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
