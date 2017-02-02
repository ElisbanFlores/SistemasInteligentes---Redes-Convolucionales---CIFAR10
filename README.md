# SistemasInteligentes---Redes-Convolucionales---CIFAR10

Modelado de Red convolucional con TensorFlow, para entrenar la base de datos CIFAR10 (dataset http://www.cs.utoronto.ca/~kriz/cifar.html )
Se ha modelado en base al tutorial (https://www.tensorflow.org/tutorials/deep_cnn/) para el mismo caso de TensorFlow

Se ha modificado el archivo cifar10.py, del codigo fuente que ofrece TensorFlow para el tutorial, en los demás archivos solo se han cambiado parámetros para el numero de iteraciones para el entrenamiento

Se ha trabajado en una PC con procesador Intel(R) Core(TM) i7-4770 CPU @ 3.40Hz con RAM 8,00GB, GPU GeForceGTX 750 Ti 1.2545GHz Memory 2.00GB, con SO Windows 10 64bits.

Se ha instalado CUDA 8, CUDANN para CUDA 8, Python 3.5.2, Numpy, TensorFlow para GPU.

Se ha entrenado el modelo con 25000 pasos en un tiempo de 2:30 horas aprox, obteniendose una precision de 0.842.

Solo hay que ejecutar
Para entrenar en GPU: python cifar10_multi_gpu_train.py
Para entrenar en CPU: python cifar10_train.py
Para Evaluar : python cifar10_train.py

El numero de pasos se puede modificar en cifar10_multi_gpu_train.py ó python cifar10_train.py (tf.app.flags.DEFINE_integer('max_steps', 25000,)

Elisban Flores Quenaya
MCC - UCSP
