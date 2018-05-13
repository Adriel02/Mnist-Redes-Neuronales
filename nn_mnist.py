import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y= one_hot(train_y,10)

valid_x, valid_y= valid_set
valid_y= one_hot(valid_y,10)

test_x,test_y= test_set
test_y= one_hot(test_y,10)
lista_entrenamiento= []
lista_validacion=[]


x = tf.placeholder("float", [None, 784])  # samples 28x28=784
y_ = tf.placeholder("float", [None, 10])  # labels 10 Clases


W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

Wh = tf.Variable(np.float32(np.random.rand(10,10))*0.1)
bh = tf.Variable(np.float32(np.random.rand(10))*0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.matmul(h, Wh) + bh
y = tf.nn.softmax(tf.matmul(h2, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)



print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
#Epoch=epoca, para ir viendo el error a medida que se ejecuta el programa
epoch=0
diferencia=100.0 #Para calcular la diferencia entre el error actual y el anterior
porcentajeError=0;
while diferencia>=porcentajeError:
    epoch+=1
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size] #Los cogemos de 20 en 20 y llamamos a la funcion train pasandoselos por parametro
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    #Errores entrenamiento


    dato_entrenamiento= sess.run(loss,feed_dict={x: batch_xs, y_: batch_ys})
    lista_entrenamiento.append(dato_entrenamiento)
    print "Epoca Entrenamiento:",epoch,"Error:",dato_entrenamiento #Imprimimos en que epica estamos y que error tiene en esa epoca
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(result, batch_ys): #Mediante el zip vamos viendo tanto el result (el resultado de la neurona como y que seria la etiqueta)
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

    #Errores Validacion

    #Llamamos a la funcion loss tambien pero ahora para el calculo del error en los datos de validacion, en vez de los datos de entrenamiento. (se los pasamos por parametro)
    dato_validacion= sess.run(loss,feed_dict={x: valid_x, y_: valid_y})
    lista_validacion.append(dato_validacion) #Unavez calculado el error en la validacion se lo pasamos a la lista
    print "Epoca Validacion:", epoch, "Error:", dato_validacion #Mostramos la epoca y el error en la validacion
    print "Porcentaje:", porcentajeError

    if epoch>1: #Tiene minimo que haber 2 valores para poder hacer una resta por eso empezamos en mayor que 1
        diferencia= lista_validacion[-2] - dato_validacion #Cogemos el error anterior y le restamos el error actual
        porcentajeError=dato_validacion*0.01;
    print "Diferencia",diferencia #Imprimimos la diferencia entre ambos
    print "----------------------------------------------------------------------------------"

#print lista_entrenamiento
#print lista_validacion

# Errores de validacion 15%
print "----------------------"
print "   Test result...     "
print "----------------------"

total = 0.0
error = 0.0
datos_test = sess.run(y, feed_dict={x: test_x}) #Llamamos a la funcion pasandole los datos de test
for b, r in zip(test_y, datos_test): #recorriendo tanto la etiqueta como el resultado vamos comprobando el maximo en ambos, si no son iguales sumamos 1 a un contador de errores
    if np.argmax(b) != np.argmax(r):
        error += 1
    total += 1 # En cada iteracion sumamos 1 al total para comprobar cuantas iteraciones hay y asi poder calcular el porcentaje de error y de acierto
fail = error / total * 100.0 #Calculo del porcentaje de error
print "Porcentaje de error: ", fail,"% y portenjate de exito", (100.0 - fail), "%" #Imprimimos por consola tanto el porcentaje de error como el de acierto, este ultimo es el total menos el del error

plot.ylabel('Errores') #Eje y se llamara Errores
plot.xlabel('Epocas') #Eje x se llamara Epocas
entrada_entrenamiento, = plot.plot(lista_entrenamiento) #Creamos una grafica pasandole la lista de entrenamiento (al pasar 1 parametro toma como eje x valores de 1..N-1
entrada_validacion, = plot.plot(lista_validacion) #Igual que en el anterior
plot.legend(handles=[entrada_entrenamiento, entrada_validacion], #Creamos una grafica donde aparezcan las 2 anteriores
            labels=['Error entrenamiento', 'Error validacion'])
plot.savefig('Grafica_mnist.png') #La guardamos en la carpeta del proyecto
# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#vplt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
