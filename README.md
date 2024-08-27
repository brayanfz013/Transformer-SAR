# Implementacion de Restormer (CVPR 2022 -- Oral) y Uformer (CVPR 2022)

Uformer Paper link: [[Arxiv]](https://arxiv.org/abs/2106.03106) [[CVPR]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)


Restormer Paper link: [[Arxiv]](https://arxiv.org/abs/2111.09881) [[CVPR]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)

## Instalacion 
Para el manejo de las dependencias de Python, se uso [Poetry](https://python-poetry.org/docs/). Este debe de ser instalado al igual que la version 3.10 de Python.

Para instalar poetry, hacerlo desde 


1. Verificar la versión de Python:
    ``` powershell
        python --version
    ```

2. Instalar Poetry:
    - Windows: 
        ``` powershell
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
        ```
    - Linux
        ``` shell 
        curl -sSL https://install.python-poetry.org | python3 -
        ```

3. Clonar el repositorio `https://github.com/brayanfz013/Transformer-SAR` ⚠️
    ``` shell
    cd Transformer-SAR
    ```
4. Inicializar el proyecto con Poetry (esto debe de hacerse en la carpeta que contenga el archivo `pyproject.toml`) ⚠️
    ``` shell
    poetry install
    poetry shell
    ```

## Dataset

Para replicar el proceso de entrenamiento del modelo es necesario modificar las rutas que que contiene las imagenes en el archivos __config.yaml__ ubicado en :

``` shell
 /pytorch_env/transformers/src/data/config/config.yaml
```
donde image-path-raw es el dataset original base para el proceso de aumento de datos

``` yaml
preprocessing:
  image-path-raw: "/home/pytorch_env/pytorch_env/transformers/src/data/raw_data"
  output: "/home/pytorch_env/pytorch_env/transformers/src/data/transformed_data/"
  train-split-percentage: 0.8
```

para realizar el aumento de la informacion es necesario ejecutar el codigo de `preprocessing.py`
de la siguiente manera:

``` shell
cd  pytorch_env/transformers/src
```
ejecuar el archivo de la siguiente manera

``` shell 
python preprocessing.py
```
las imagenes se encuentran en el siguiente enlace de google drive:

imagenes enlace : [[Google Drive]](https://drive.google.com/drive/folders/1aRyshK11brNl0SFul3jSM5RPClomo55e?usp=drive_link)

las cuales estan divididas en las siguientes carpetas:

| Carpeta | Contenido | Formato de archivos | Observaciones |
|---|---|---|---|
| 01_Imagenes Restauradas |  |  |  |
|   | Ground Truth | Imágenes originales | Descargadas de satélite |
|   | Noisy | Imágenes con ruido artificial |  |
|   | Modelo Restormer |  |  |
|   |   | CharbonierLoss |  |
|   |   |   | Imágenes: Número_image_NumeroModelo.jpg |
|   |   |   | Modelos: Listado de modelos |
|   |   | L1Loss_0_99 |  |
|   |   |   | Imágenes: Número_image_NumeroModelo.jpg |
|   |   |   | Modelos: Listado de modelos |
|   |   | PSNR |  |
|   |   |   | Imágenes: Número_image_NumeroModelo.jpg |
|   |   |   | Modelos: Listado de modelos |
|   | Modelo Uformer |  |  |
|   |   | Imágenes Restauradas Uformer |  |
|   |   | Modelos | Modelo único |
| Noisy_val | Imágenes descargadas de satélite |  | Validación por SwinIR |
| Transformer_data | Datos aumentados |  |  |
| raw_data | Imágenes descargadas de satélite |  |  |
| validation_data_report | Imágenes de validación |  | Entregadas originalmente |

![alt text](<Mapa Google Drive.png>)

## Entrenamiento e Inferencia

```shell
cd pytorch_env/transformers/src/notebooks/
```

Para entrenar nuevamente un modelo o realizar una inferencia/prediccion leer contenido del notebook. `VIT.ipynb`


## Citacion


Si quiere citar el paper del uformer  y el Restomer considere:

```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhendong and Cun, Xiaodong and Bao, Jianmin and Zhou, Wengang and Liu, Jianzhuang and Li, Houqiang},
    title     = {Uformer: A General U-Shaped Transformer for Image Restoration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17683-17693}
}
```
```
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```


