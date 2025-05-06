# 🐱❓🐶 Classificador de Gatos vs Cães com Transfer Learning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

<div align="center">
  <img src="https://github.com/yourusername/cats-vs-dogs-transfer-learning/raw/main/images/demo.gif" width="600" alt="Demonstração do classificador">
</div>

## 📝 Sumário
- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Dataset](#-dataset)
- [Arquitetura](#-arquitetura-do-modelo)
- [Resultados](#-resultados)
- [Como Usar](#-como-usar)
- [Exemplos](#-exemplos)
- [Licença](#-licença)

## 🌟 Visão Geral
Classificador de imagens que diferencia entre gatos e cães utilizando Transfer Learning com o modelo VGG16 pré-treinado no ImageNet. Atingimos **96% de acurácia** com pouco tempo de treinamento.
Para ver o código a partir do google colab acesse o link: https://colab.research.google.com/drive/1EjQG1JMQoW7lfcnbkAx8vAcjy8XVKNFX#scrollTo=tgAMaEbs7wDF

## ✨ Funcionalidades
- ✅ Classificação binária (gato vs cachorro)
- ✅ Data augmentation para melhor generalização
- ✅ Visualização do processo de treinamento
- ✅ Função pronta para predições em novas imagens
- ✅ Modelo leve e eficiente

## 💻 Tecnologias
- **TensorFlow 2.x** - Framework de deep learning
- **Keras** - API de alto nível para TensorFlow
- **VGG16** - Arquitetura de CNN pré-treinada
- **OpenCV** - Pré-processamento de imagens
- **Matplotlib** - Visualização dos resultados

## 📁 Dataset
Utilizamos o [Microsoft Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765):
- 25,000 imagens (12,500 de cada classe)
- Dividido automaticamente em:
  - Treino (80%)
  - Validação (20%)

## 🧠 Arquitetura do Modelo
```python
model = Sequential([
    VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

